/**
 * Chatbot widget for MkDocs Material
 */
document$.subscribe(function () {
  // Check if widget already exists (to prevent duplicates during navigation)
  if (document.getElementById("chatbot-widget")) {
    return;
  }

  // Create the chatbot UI
  const chatbotElement = document.createElement("div");
  chatbotElement.id = "chatbot-widget";
  chatbotElement.className = "chatbot-widget";

  // Define the HTML structure
  chatbotElement.innerHTML = `
      <button id="chatbot-toggle" class="chatbot-toggle">
        <span>Chat</span>
      </button>
      <div id="chatbot-container" class="chatbot-container" style="display: none;">
        <div class="chatbot-header">
          <span>Documentation Assistant</span>
          <button id="chatbot-close">×</button>
        </div>
        <div class="chatbot-body" id="chatbot-messages">
          <div class="chatbot-message bot">
            <div class="chatbot-message-content">
              Hello! I'm your documentation assistant. How can I help you today?
            </div>
          </div>
        </div>
        <div class="chatbot-input-container">
          <input 
            type="text" 
            id="chatbot-input" 
            placeholder="Type your message..." 
            aria-label="Type your message"
          >
          <button id="chatbot-send">Send</button>
        </div>
      </div>
    `;

  // Add the widget to the page
  document.body.appendChild(chatbotElement);

  // Cache DOM elements
  const chatbotToggle = document.getElementById("chatbot-toggle");
  const chatbotContainer = document.getElementById("chatbot-container");
  const chatbotClose = document.getElementById("chatbot-close");
  const chatbotInput = document.getElementById("chatbot-input");
  const chatbotSend = document.getElementById("chatbot-send");
  const chatbotMessages = document.getElementById("chatbot-messages");

  // Toggle the chatbot visibility
  chatbotToggle.addEventListener("click", function () {
    const isVisible = chatbotContainer.style.display !== "none";
    chatbotContainer.style.display = isVisible ? "none" : "flex";

    // Focus the input field when opening
    if (!isVisible) {
      setTimeout(() => chatbotInput.focus(), 100);
    }
  });

  // Close the chatbot
  chatbotClose.addEventListener("click", function () {
    chatbotContainer.style.display = "none";
  });

  // Send message function
  function sendMessage() {
    const message = chatbotInput.value.trim();
    if (!message) return;

    // Add user message to the chat
    addMessage(message, "user");

    // Clear the input field
    chatbotInput.value = "";

    // Show typing indicator
    const typingIndicator = document.createElement("div");
    typingIndicator.className = "chatbot-message bot typing";
    typingIndicator.innerHTML = `
        <div class="chatbot-message-content">
          <div class="typing-indicator">
            <span></span><span></span><span></span>
          </div>
        </div>
      `;
    chatbotMessages.appendChild(typingIndicator);

    // Scroll to bottom
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;

    // Call the API
    fetch("http://localhost:8080/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Remove typing indicator
        if (typingIndicator.parentNode) {
          typingIndicator.parentNode.removeChild(typingIndicator);
        }

        // Add bot response
        addMessage(data.response, "bot");
      })
      .catch((error) => {
        // Remove typing indicator
        if (typingIndicator.parentNode) {
          typingIndicator.parentNode.removeChild(typingIndicator);
        }

        // Add error message
        addMessage(
          "Sorry, I encountered an error. Please try again later.",
          "bot error"
        );
        console.error("Error:", error);
      });
  }

  // Add message to the chat
  function addMessage(text, sender) {
    const messageElement = document.createElement("div");
    messageElement.className = `chatbot-message ${sender}`;
    messageElement.innerHTML = `
        <div class="chatbot-message-content">
          ${text}
        </div>
      `;

    chatbotMessages.appendChild(messageElement);

    // Scroll to the new message
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
  }

  // Send message on button click
  chatbotSend.addEventListener("click", sendMessage);

  // Send message on Enter key
  chatbotInput.addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
      sendMessage();
    }
  });

  // Focus input when clicking anywhere in the message area
  chatbotMessages.addEventListener("click", function () {
    chatbotInput.focus();
  });
});
