#!/bin/sh

# Exit immediately if a command exits with a non-zero status
set -e
# Usage documentation:
# This script installs and configures the NVIDIA NGC CLI tool.
#
# Arguments:
#   --ngc-api-key <key>           : Your NGC API key (mandatory if login is enabled).
#   --ngc-org <organization>      : Your NGC organization (mandatory if login is enabled).
#   --ngc-team <team>             : Your NGC team (mandatory if login is enabled).
#   --installation-folder <path>  : Directory where the NGC CLI will be installed (mandatory).
#   --no-ngc-login                : Flag to bypass NGC login configuration.
#
# Example usage:
#   ./setup-ngc-cli.sh --ngc-api-key YOUR_API_KEY --ngc-org YOUR_ORG --ngc-team YOUR_TEAM --installation-folder /path/to/install
#   ./setup-ngc-cli.sh --installation-folder /path/to/install --no-ngc-login

# Default value for NGC login
NGC_LOGIN="True"

# Parse input arguments
while [ $# -gt 0 ]; do
    KEY="$1"
    case $KEY in
        --ngc-api-key)
            NGC_KEY="$2"
            shift 2;;
        --ngc-org)
            NGC_ORG="$2"
            shift 2;;
        --ngc-team)
            NGC_TEAM="$2"
            shift 2;;
        --installation-folder)
            INSTALLATION_DIR="$2"
            shift 2;;
        --no-ngc-login)
            NGC_LOGIN="False"
            shift;;
        --*=|-*)
            echo "Error: Unsupported keyword ${KEY}" >&2
            exit 1 ;;
        *)
            echo "Error: Unsupported positional argument ${KEY}" >&2
            exit 1 ;;
    esac
done

# Validate required arguments
if [ -z "${INSTALLATION_DIR}" ]; then
    echo "Error: --installation-folder must be set" >&2
    exit 1
fi

if [ "$NGC_LOGIN" = "True" ] && { [ -z "${NGC_KEY}" ] || [ -z "${NGC_ORG}" ] || [ -z "${NGC_TEAM}" ]; }; then
    echo "Error: The arguments --ngc-api-key, --ngc-org, and --ngc-team must be specified. To bypass NGC login, use the --no-ngc-login flag." >&2
    exit 1
fi

# Create installation directory and move into it
mkdir -p "${INSTALLATION_DIR}"
cd "${INSTALLATION_DIR}"

# Select appropriate version of NGC CLI
FILE_TO_DOWNLOAD="ngccli_linux.zip"
if SYS_ARCH=$(./ci/scripts/get_system_arch.sh); then
    echo "System architecture: ${SYS_ARCH}"
    if [ "${SYS_ARCH}" = "arm64" ]; then
        FILE_TO_DOWNLOAD="ngccli_arm64.zip"
    fi
else
    echo "Error determining system architecture"
    exit 1
fi

# Install required packages
apt update > /dev/null && apt install -y wget unzip > /dev/null

# Download and extract the NGC CLI
wget --content-disposition https://ngc.nvidia.com/downloads/${FILE_TO_DOWNLOAD} -O ngccli.zip
unzip -q -o ngccli.zip
chmod u+x ngc-cli/ngc

# Verify integrity of the files if md5 checksum file exists
find ngc-cli/ -type f -exec md5sum {} + | LC_ALL=C sort | md5sum -c ngc-cli.md5

./ngc-cli/ngc --version

# Configure NGC login if needed
if [ "$NGC_LOGIN" = "True" ]; then
    printf "%s\n" "${NGC_KEY}" json "${NGC_ORG}" "${NGC_TEAM}" no-ace | ./ngc-cli/ngc config set
fi
