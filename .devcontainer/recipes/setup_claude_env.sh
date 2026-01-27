#!/bin/bash
# setup_env.sh - Environment setup for Claude Code with NVIDIA Hosted Endpoint
#
# This script sets up Claude Code to use NVIDIA's hosted Claude API via LiteLLM proxy.
# It handles everything: API key setup, LiteLLM installation, and proxy startup.
#
# Usage:
#   source scripts/setup_env.sh           # Basic setup + start proxy
#   source scripts/setup_env.sh --venv    # Create/activate virtual environment first
#   source scripts/setup_env.sh --restart # Force restart proxy (e.g., after changing API key)
#   source scripts/setup_env.sh --verbose # Show detailed install output
#
# What this script does:
#   1. Installs uv (fast Python package manager) if not present
#   2. Installs Claude Code CLI (via native installer) if not present
#   3. Reads NVIDIA API key from ~/.nvidia-api-key or prompts and saves it
#   4. Installs LiteLLM proxy using uv
#   5. Starts the LiteLLM proxy server in the background
#   6. Configures Claude Code via ~/.claude/settings.json (proper method)
#   7. Skips Claude onboarding via ~/.claude.json
#
# Prerequisites:
#   - NVIDIA API key from https://inference.nvidia.com/key-management
#   - Basic system tools: curl, lsof, python3 (usually pre-installed)
#
# API Key Storage:
#   The script reads your NVIDIA API key from ~/.nvidia-api-key
#   If the file doesn't exist or is empty, it will prompt you for the key
#   and save it securely to that file with restricted permissions (600)

# ============================================================================
# Check: Must be sourced, not executed
# ============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo ""
    echo "❌ ERROR: This script must be SOURCED, not executed directly."
    echo ""
    echo "   Wrong:   ./scripts/setup_env.sh"
    echo "   Wrong:   bash scripts/setup_env.sh"
    echo "   Correct: source scripts/setup_env.sh"
    echo ""
    echo "   Sourcing exports environment variables to your current shell."
    exit 1
fi

# Safe error handling for sourced scripts
# Use 'return' instead of 'exit' to avoid killing the shell
safe_exit() {
    return 1
}

# Parse arguments
FORCE_RESTART=false
USE_VENV=false
VERBOSE=false
for arg in "$@"; do
    case $arg in
        --restart) FORCE_RESTART=true ;;
        --venv) USE_VENV=true ;;
        --verbose) VERBOSE=true ;;
    esac
done

# Log file for install output
INSTALL_LOG="/tmp/claude_nvidia_setup.log"
echo "" > "$INSTALL_LOG"

# ============================================================================
# Helper Functions
# ============================================================================

# Spinner animation
spin() {
    local pid=$1
    local msg=$2
    local spinchars='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local i=0

    # Hide cursor
    tput civis 2>/dev/null || true

    while kill -0 "$pid" 2>/dev/null; do
        # Use arithmetic expansion for shell compatibility (bash and zsh)
        local idx=$((i % ${#spinchars}))
        printf "\r   %s %s" "${spinchars:$idx:1}" "$msg"
        i=$((i + 1))
        sleep 0.1
    done

    # Show cursor
    tput cnorm 2>/dev/null || true

    # Wait for process and get exit code
    wait "$pid"
    return $?
}

# Run command with spinner (suppresses output unless verbose)
run_with_spinner() {
    local msg=$1
    shift

    if [ "$VERBOSE" = true ]; then
        echo "   $msg"
        "$@"
        return $?
    fi

    # Run in background, capture output to log
    "$@" >> "$INSTALL_LOG" 2>&1 &
    local pid=$!

    if spin "$pid" "$msg"; then
        printf "\r   ✓ %-60s\n" "$msg"
        return 0
    else
        printf "\r   ❌ %-60s\n" "$msg"
        echo "      See log: $INSTALL_LOG"
        return 1
    fi
}

# Print status line
status() {
    printf "   ✓ %s\n" "$1"
}

# Print warning
warn() {
    printf "   ⚠️  %s\n" "$1"
}

# Print error
error() {
    printf "   ❌ %s\n" "$1"
}

# ============================================================================
# Header
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║       Claude Code + NVIDIA Inference - Environment Setup          ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# Check System Dependencies
# ============================================================================
echo "📦 System Dependencies"
echo "   ─────────────────────────────────────────────────────────────────"

# Check basic commands
MISSING_CMDS=""
for cmd in curl lsof python3; do
    if ! command -v $cmd &> /dev/null; then
        MISSING_CMDS="$MISSING_CMDS $cmd"
    fi
done

if [ -n "$MISSING_CMDS" ]; then
    error "Missing required commands:$MISSING_CMDS"
    echo "      Install with: sudo apt install$MISSING_CMDS"
    return 1
fi
status "curl, lsof, python3"

# ============================================================================
# Install uv (Python package manager)
# ============================================================================
if ! command -v uv &> /dev/null; then
    run_with_spinner "Installing uv (Python package manager)..." \
        bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'

    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if command -v uv &> /dev/null; then
    PKG_MANAGER="uv"
    PKG_INSTALL="uv pip install"
    status "uv $(uv --version 2>/dev/null | head -1 || echo '')"
elif command -v pip &> /dev/null; then
    PKG_MANAGER="pip"
    PKG_INSTALL="pip install"
    warn "uv not available, using pip"
else
    error "Neither 'uv' nor 'pip' available"
    return 1
fi

# ============================================================================
# Claude Code CLI Setup (Native Installer)
# ============================================================================
echo ""
echo "🤖 Claude Code CLI"
echo "   ─────────────────────────────────────────────────────────────────"

# Add common installation paths
export PATH="$HOME/.local/bin:$PATH"

if command -v claude &> /dev/null; then
    CLAUDE_VERSION=$(claude --version 2>/dev/null || echo "unknown")
    status "Claude CLI $CLAUDE_VERSION"
else
    # Use native installer (npm installation is deprecated)
    run_with_spinner "Installing Claude Code CLI (native installer)..." \
        bash -c 'curl -fsSL https://claude.ai/install.sh | bash'

    # Update PATH to include installation location
    export PATH="$HOME/.local/bin:$PATH"

    # Verify installation
    if command -v claude &> /dev/null; then
        status "Claude CLI installed"
    else
        warn "Installation completed but 'claude' not in PATH"
        echo "      Try: export PATH=\"\$HOME/.local/bin:\$PATH\""
        echo "      Or run: claude install"
    fi
fi

# Optional: Create virtual environment
if [ "$USE_VENV" = true ]; then
    echo ""
    echo "📦 Virtual Environment"
    echo "   ─────────────────────────────────────────────────────────────────"
    if [ ! -d "venv" ]; then
        run_with_spinner "Creating virtual environment..." \
            python3 -m venv venv
    fi
    # shellcheck disable=SC1091
    source venv/bin/activate
    status "Activated: $(which python)"
fi

# ============================================================================
# NVIDIA API Key Setup
# ============================================================================
echo ""
echo "🔑 NVIDIA API Key"
echo "   ─────────────────────────────────────────────────────────────────"

# Define the API key file
API_KEY_FILE="$HOME/.nvidia-api-key"

# Try to read the key from the file
NVIDIA_API_KEY=""
if [ -f "$API_KEY_FILE" ]; then
    # Read the key from file, removing any trailing whitespace
    NVIDIA_API_KEY=$(cat "$API_KEY_FILE" 2>/dev/null | tr -d '\n\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

    if [ -n "$NVIDIA_API_KEY" ]; then
        # Validate the key format
        if [[ ! "$NVIDIA_API_KEY" =~ ^sk- ]]; then
            error "API key found in $API_KEY_FILE but format is invalid"
            error "Key found: ${NVIDIA_API_KEY:0:30}..."
            error "Key must start with 'sk-' to work. Clearing invalid key."
            NVIDIA_API_KEY=""
        else
            status "Loaded NVIDIA_API_KEY from $API_KEY_FILE (${NVIDIA_API_KEY:0:7}...)"
        fi
    fi
fi

# If no valid key found, prompt the user
if [ -z "$NVIDIA_API_KEY" ]; then
    if [ -t 0 ]; then  # Interactive terminal
        echo "   Get your key: https://inference.nvidia.com/key-management"
        echo "   Key format: sk-xxxxxxxxxxxxxxxxxxxxxxxx"
        echo ""
        # Use shell-compatible read syntax (works in both bash and zsh)
        if [ -n "${ZSH_VERSION:-}" ]; then
            read -s "?   Enter your NVIDIA API key: " NVIDIA_API_KEY
        else
            read -sp "   Enter your NVIDIA API key: " NVIDIA_API_KEY
        fi
        echo ""

        if [ -n "$NVIDIA_API_KEY" ]; then
            # Validate user-entered key format
            if [[ ! "$NVIDIA_API_KEY" =~ ^sk- ]]; then
                error "Invalid key format: must start with 'sk-'"
                error "You entered: ${NVIDIA_API_KEY:0:20}..."
                echo "      Please try again with a valid NVIDIA API key."
                NVIDIA_API_KEY=""
                return 1
            fi
            # Save to the API key file
            echo "$NVIDIA_API_KEY" > "$API_KEY_FILE"
            chmod 600 "$API_KEY_FILE"  # Secure file permissions (owner read/write only)
            status "Saved to $API_KEY_FILE"
        fi
    else
        error "NVIDIA_API_KEY not found in $API_KEY_FILE (non-interactive mode)"
        echo "      Create the file: echo 'your-key' > ~/.nvidia-api-key"
        echo "      Make sure the key starts with 'sk-'"
        return 1
    fi
fi

# Final validation: key MUST start with "sk-" to work
if [ -n "$NVIDIA_API_KEY" ]; then
    if [[ ! "$NVIDIA_API_KEY" =~ ^sk- ]]; then
        error "Invalid API key format: must start with 'sk-'"
        error "Found key starting with: ${NVIDIA_API_KEY:0:20}..."
        echo "      Please set a valid NVIDIA API key."
        echo "      Get your key: https://inference.nvidia.com/key-management"
        return 1
    fi
    export NVIDIA_API_KEY
    status "Key configured (${NVIDIA_API_KEY:0:7}...)"
else
    error "NVIDIA_API_KEY is not set"
    return 1
fi

# ============================================================================
# LiteLLM Proxy Setup
# ============================================================================
echo ""
echo "🔄 LiteLLM Proxy"
echo "   ─────────────────────────────────────────────────────────────────"

# Check if LiteLLM is installed and working
LITELLM_NEEDS_INSTALL=false
if ! command -v litellm &> /dev/null; then
    if [ -f "venv/bin/litellm" ] && [ "$USE_VENV" = false ]; then
        warn "LiteLLM found in venv/ but venv not activated"
        echo "      Re-run with: source scripts/setup_env.sh --venv"
        return 1
    fi
    LITELLM_NEEDS_INSTALL=true
else
    # LiteLLM exists, but check if proxy dependencies are installed
    if ! python3 -c "import backoff" 2>/dev/null; then
        warn "LiteLLM installed but missing proxy dependencies"
        LITELLM_NEEDS_INSTALL=true
    fi
fi

if [ "$LITELLM_NEEDS_INSTALL" = true ]; then
    # Use requirements.txt if available (more reliable than extras)
    if [ -f "requirements.txt" ]; then
        run_with_spinner "Installing LiteLLM from requirements.txt..." \
            $PKG_INSTALL -r requirements.txt
    else
        # Fallback: install litellm[proxy] directly
        run_with_spinner "Installing LiteLLM..." \
            $PKG_INSTALL 'litellm[proxy]'
    fi
    status "LiteLLM installed"
else
    status "LiteLLM ready"
fi

# Check if config file exists
if [ ! -f ".devcontainer/recipes/litellm_config.yaml" ]; then
    error "litellm_config.yaml not found"
    return 1
fi

# ============================================================================
# Start LiteLLM Proxy Server
# ============================================================================
echo ""
echo "🚀 Starting Proxy"
echo "   ─────────────────────────────────────────────────────────────────"

# Function to start proxy with spinner
start_proxy_with_spinner() {
    # Backup old log
    [ -f /tmp/litellm_proxy.log ] && mv /tmp/litellm_proxy.log /tmp/litellm_proxy.log.bak 2>/dev/null || true

    # Start LiteLLM in background
    nohup litellm --config .devcontainer/recipes/litellm_config.yaml --port 4000 > /tmp/litellm_proxy.log 2>&1 &
    local PROXY_PID=$!

    # Spinner while waiting for health
    local spinchars='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local i=0
    local msg="Starting proxy on port 4000..."

    tput civis 2>/dev/null || true

    local count=0
    while [ $count -lt 30 ]; do
        if curl -s http://localhost:4000/health &> /dev/null; then
            tput cnorm 2>/dev/null || true
            printf "\r   ✓ %-60s\n" "Proxy running (PID: $PROXY_PID)"
            return 0
        fi

        if ! kill -0 $PROXY_PID 2>/dev/null; then
            tput cnorm 2>/dev/null || true
            printf "\r   ❌ %-60s\n" "Proxy failed to start"
            echo "      Check logs: cat /tmp/litellm_proxy.log"
            return 1
        fi

        # Use arithmetic expansion for shell compatibility (bash and zsh)
        local idx=$((i % ${#spinchars}))
        printf "\r   %s %s" "${spinchars:$idx:1}" "$msg"
        i=$((i + 1))
        count=$((count + 1))
        sleep 1
    done

    tput cnorm 2>/dev/null || true
    printf "\r   ❌ %-60s\n" "Proxy timeout after 30 seconds"
    return 1
}

# Check if proxy is already running
if lsof -i :4000 &> /dev/null; then
    if curl -s http://localhost:4000/health &> /dev/null; then
        if [ "$FORCE_RESTART" = true ]; then
            printf "   %s %s" "⠋" "Restarting proxy..."
            lsof -ti :4000 | xargs kill 2>/dev/null || true
            sleep 2
            [ "$(lsof -i :4000 2>/dev/null)" ] && lsof -ti :4000 | xargs kill -9 2>/dev/null || true
            sleep 1
            printf "\r   ✓ %-60s\n" "Stopped old proxy"
            start_proxy_with_spinner || return 1
        else
            # Proxy is running - ask if user wants to restart (configs may have changed)
            if [ -t 0 ]; then  # Interactive terminal
                echo "   Proxy already running on port 4000"
                echo "   Re-sourcing may have updated configurations."
                echo ""
                # Use shell-compatible read syntax
                RESTART_CHOICE=""
                if [ -n "${ZSH_VERSION:-}" ]; then
                    read "?   Restart proxy to apply changes? [y/N]: " RESTART_CHOICE
                else
                    read -p "   Restart proxy to apply changes? [y/N]: " RESTART_CHOICE
                fi

                if [[ "$RESTART_CHOICE" =~ ^[Yy]$ ]]; then
                    printf "   %s %s" "⠋" "Restarting proxy..."
                    lsof -ti :4000 | xargs kill 2>/dev/null || true
                    sleep 2
                    [ "$(lsof -i :4000 2>/dev/null)" ] && lsof -ti :4000 | xargs kill -9 2>/dev/null || true
                    sleep 1
                    printf "\r   ✓ %-60s\n" "Stopped old proxy"
                    start_proxy_with_spinner || return 1
                else
                    status "Keeping existing proxy running"
                fi
            else
                # Non-interactive: just report it's running
                status "Proxy already running on port 4000"
            fi
        fi
    else
        error "Port 4000 in use by another process"
        echo "      Kill it: lsof -ti :4000 | xargs kill -9"
        return 1
    fi
else
    start_proxy_with_spinner || return 1
fi

# ============================================================================
# Configure Claude Code (via settings.json)
# ============================================================================
echo ""
echo "⚙️  Claude Code Config"
echo "   ─────────────────────────────────────────────────────────────────"

# Create Claude config directory
mkdir -p "$HOME/.claude"

# Create/update ~/.claude/settings.json with proxy configuration
CLAUDE_SETTINGS="$HOME/.claude/settings.json"
if [ -f "$CLAUDE_SETTINGS" ]; then
    # Backup existing settings
    cp "$CLAUDE_SETTINGS" "$CLAUDE_SETTINGS.bak"

    # Check if we need to update (using python for JSON handling)
    python3 << EOF
import json
import sys

settings_file = "$CLAUDE_SETTINGS"
try:
    with open(settings_file, 'r') as f:
        settings = json.load(f)
except (json.JSONDecodeError, FileNotFoundError):
    settings = {}

# Ensure env section exists
if 'env' not in settings:
    settings['env'] = {}

# Update proxy settings
settings['env']['ANTHROPIC_API_KEY'] = 'sk-litellm-local-dev'
settings['env']['ANTHROPIC_BASE_URL'] = 'http://localhost:4000'

with open(settings_file, 'w') as f:
    json.dump(settings, f, indent=2)
EOF
    status "Updated $CLAUDE_SETTINGS"
else
    # Create new settings.json
    cat > "$CLAUDE_SETTINGS" << 'EOF'
{
  "env": {
    "ANTHROPIC_API_KEY": "sk-litellm-local-dev",
    "ANTHROPIC_BASE_URL": "http://localhost:4000"
  }
}
EOF
    status "Created $CLAUDE_SETTINGS"
fi

# Create/update ~/.claude.json to skip onboarding
# When using self-hosted API, we must skip onboarding to avoid sign-in prompts
CLAUDE_JSON="$HOME/.claude.json"
if [ ! -f "$CLAUDE_JSON" ]; then
    echo '{"hasCompletedOnboarding": true}' > "$CLAUDE_JSON"
    status "Created $CLAUDE_JSON (skip onboarding for self-hosted API)"
else
    # Always force hasCompletedOnboarding to true when using self-hosted API
    # This ensures onboarding is skipped even if it was previously set to false
    python3 << EOF
import json
import sys
config_file = "$CLAUDE_JSON"
try:
    with open(config_file, 'r') as f:
        config = json.load(f)
except (json.JSONDecodeError, FileNotFoundError):
    config = {}

# Force onboarding to be completed when using self-hosted API
old_value = config.get('hasCompletedOnboarding', None)
config['hasCompletedOnboarding'] = True

with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

# Only show message if we changed the value
if old_value != True:
    import sys
    if old_value is None:
        sys.exit(1)  # Was missing, now added
    elif old_value == False:
        sys.exit(2)  # Was false, now true
    else:
        sys.exit(0)  # Was already true
else:
    sys.exit(0)  # Was already true
EOF
    exit_code=$?
    case $exit_code in
        1)
            status "Added hasCompletedOnboarding to $CLAUDE_JSON (skip onboarding)"
            ;;
        2)
            status "Updated $CLAUDE_JSON: forced hasCompletedOnboarding=true (skip onboarding for self-hosted API)"
            ;;
        0)
            status "Onboarding already skipped in $CLAUDE_JSON"
            ;;
    esac
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                        ✓ Setup Complete!                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "   🟢 Proxy: http://localhost:4000"
echo "   🔧 Config: ~/.claude/settings.json"
echo ""
echo "   Run Claude Code:"
echo ""
echo "      claude"
echo ""
echo "─────────────────────────────────────────────────────────────────────"
echo "   Proxy logs: /tmp/litellm_proxy.log"
echo "   Stop proxy: lsof -ti :4000 | xargs kill"
echo "   Restart:    source scripts/setup_env.sh --restart"
echo "─────────────────────────────────────────────────────────────────────"
echo ""
