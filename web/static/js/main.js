let currentButton = null;

function showFeedbackPane(button) {
    const feedbackPane = document.getElementById('feedbackPane');
    const chatPane = document.getElementById('chatPane');

    chatPane.classList.add('hidden');
    feedbackPane.classList.remove('hidden');

    // Highlight the Add Feedback button
    button.classList.add('highlight-button');

    // Store the reference to the current button
    currentButton = button;
}

document.getElementById('submitFeedback').addEventListener('click', function() {
    const form = document.getElementById('feedbackForm');
    
    var urlParams = new URLSearchParams(window.location.search);
    var params = {};
    for (const [key, value] of urlParams.entries()) {
        params[key] = value;
    }

    // Get the button element by its class name
    var feedbackButton = document.querySelector('.feedbackButton.highlight-button');

    // Get the text inside the button
    var buttonText = feedbackButton.id;

    // alert(buttonText); // Outputs: Add Feedback


        
    const feedback = {
        form: form.innerHTML,
        insight:buttonText,
        ratings: {
            question1: form.querySelector('input[name="question1"]:checked')?.value,
            question2: form.querySelector('input[name="question2"]:checked')?.value,
            question3: form.querySelector('input[name="question3"]:checked')?.value
        },
        comment: form.querySelector('textarea[name="comment"]').value,
        ...params
    };

    fetch('/submit-feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(feedback)
    })
    .then(response => response.json())
    .then(data => {
        console.log('Feedback submitted successfully:', data);

        // Clear form and switch panes
        form.reset();
        feedbackPane.classList.add('hidden');
        chatPane.classList.remove('hidden');

        // Disable the Add Feedback button, change its text, and remove highlight
        if (currentButton) {
            currentButton.textContent = 'Feedback Submitted';
            currentButton.disabled = true;
            currentButton.classList.remove('highlight-button');
            currentButton.classList.add('submitted-button');
        }

        // Reset currentButton to null
        currentButton = null;
    })
    .catch(error => {
        console.error('Error submitting feedback:', error);
    });
});
// Function to open a specific tab
function openTab(evt, tabName) {
    var i, tabcontent, tablinks;

    // Hide all tab contents
    tabcontent = document.getElementsByClassName("tabcontent-main");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }

    // Remove active class from all tabs
    tablinks = document.getElementsByClassName("tablink");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    // Show the current tab and add an "active" class to the button that opened the tab
    document.getElementById(tabName).style.display = "block";
    if (evt) evt.currentTarget.className += " active";
}



function editMetadata() {
    // Get metadata element and text area
    const metadataBlob = document.getElementById('meta-data');
    const metadataEdit = document.getElementById('meta-data-edit');
    const editButton = document.getElementById('editMetadataButton');
    const saveButton = document.getElementById('saveMetadataButton');

    // Show text area and hide metadata blob
    metadataEdit.style.display = 'block';
    metadataEdit.value = metadataBlob.innerText; // Set text area value to current metadata
    metadataBlob.style.display = 'none';
    editButton.style.display = 'none';
    saveButton.style.display = 'block';
}

function saveMetadata() {
    // Get metadata element and text area
    const metadataBlob = document.getElementById('meta-data');
    const metadataEdit = document.getElementById('meta-data-edit');
    const editButton = document.getElementById('editMetadataButton');
    const saveButton = document.getElementById('saveMetadataButton');

    // Save edited metadata
    metadataBlob.innerText = metadataEdit.value;

    // Hide text area and show metadata blob
    metadataEdit.style.display = 'none';
    metadataBlob.style.display = 'block';
    editButton.style.display = 'block';
    saveButton.style.display = 'none';

    // Optionally, send the updated metadata to the server here
    // fetch('/update-metadata', { ... });
}
function updateTable() {
    // Your existing logic to clean data goes here...

    // Hide the warning message and show the success message
    document.getElementById('warning-message').style.display = 'none';
    document.getElementById('success-message').style.display = 'block';
 document.getElementById('clean-butt').style.display = 'none';
  document.getElementById('table-frame').style.display = 'block';
    // Optionally, you might want to update the table here as well
    // document.getElementById('table-frame').innerHTML = updatedTableHTML;
}






function removeLoadingBar() {
            const messagesContainer = document.getElementById('chatbotMessages');
            const loadingBarContainer = document.getElementById('loadingBarContainer');
            if (loadingBarContainer) {
                messagesContainer.removeChild(loadingBarContainer);
            }
}
// Function to add the loading bar with text
function addLoadingBar(text) {
    const container = document.getElementById('chatbotMessages');
    const loadingBarContainer = document.createElement('div');
    loadingBarContainer.id = 'loadingBarContainer';
    
    const loadingText = document.createElement('div');
    
    loadingText.id = 'loadingText';
    loadingText.textContent = text;

    const loadingBar = document.createElement('div');
    loadingBar.id = 'loadingBar';

    const loadingBarInner = document.createElement('div');
    loadingBarInner.id = 'loadingBarInner';

    loadingBar.appendChild(loadingBarInner);
    loadingBarContainer.appendChild(loadingText);
    loadingBarContainer.appendChild(loadingBar);
    container.appendChild(loadingBarContainer);
    // container.scrollTop = container.scrollHeight;
}


// Function to add a single question bubble
// function addQuestionBubble(container, questionText) {
//     const questionBubble = document.createElement('div');
//     questionBubble.classList.add('question-oval');
//     questionBubble.textContent = questionText;
//     questionBubble.onclick = () => populateInput(questionText);

//     // Append the question bubble to the container
//     container.appendChild(questionBubble);
// }

function addLuckyBubble(questionText) {
    const container = document.getElementById('chatbotMessages');
   

    const questionBubble = document.createElement('div');
    questionBubble.classList.add('question-oval-lucky');
    questionBubble.textContent = questionText;
    questionBubble.onclick = () => addLuckyInsight(questionText);

    // Append the question bubble to the container
    container.appendChild(questionBubble);
}

function addLuckyInsight(question) {

    removeBubbles()
    add_user_message("I Feel Lucky.")
    // Get the current URL's query parameters
    const queryParams = new URLSearchParams(window.location.search);
    
    // Convert the query parameters to an object
    const paramsObject = {};
    queryParams.forEach((value, key) => {
        paramsObject[key] = value;
    });

   
    addLoadingBar("Getting a new Insight (30-60 seconds)")
    fetch('/main_add_insight_lucky', {
        method: 'POST', // Use POST to send data
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(paramsObject)
    })
        .then(response => response.json())
        .then(data => {
            if (!data.isValid) {
                removeLoadingBar()
                addMessage(`Invalid: ${data.explanation}`);
                addQuestionBubbles(data.questions)
                
                
                return; // Exit the function if data is not valid
            }
            add_user_message(`${data.question}`)
            addMessage(`Insight Card (${data.insight_id}) Added to Dashboard`);
            // Populate Insights 
            const insightBlobs = document.getElementById('insight-blobs');
            insightBlobs.innerHTML = `${data.insight} ${insightBlobs.innerHTML}`;
            
           
            addQuestionBubbles(data.questions)
            removeLoadingBar()
            
        });
        
}

function removeBubbles(){
    const messagesContainer = document.getElementById('chatbotMessages');

    const existingBubbles2 = messagesContainer.querySelectorAll('.question-oval-lucky');
    existingBubbles2.forEach(bubble => bubble.remove());
    // Remove all existing questionBubble elements
    const existingBubbles = messagesContainer.querySelectorAll('.question-oval');
    existingBubbles.forEach(bubble => bubble.remove());

}

function askFollowup(button) {
    console.log("Button ID:", button.id);
    insight_dict_id = button.id.split('-')[1];

    let followup_question = document.getElementById(`followup-question-${insight_dict_id}`).textContent;

    send_bubble(followup_question)
}

function addQuestionBubbles(questions) {
    addLuckyBubble("I Feel Lucky. Get a new Insight.")
    const messagesContainer = document.getElementById('chatbotMessages');
    const questionBubbles = document.createElement('div');

    questionBubbles.classList.add('recommended-questions');

    questions.forEach(question => {
        const questionBubble = document.createElement('div');
        questionBubble.classList.add('question-oval');
        questionBubble.textContent = question;
        questionBubble.onclick = () => send_bubble(question);
        questionBubbles.appendChild(questionBubble);
    });

    messagesContainer.appendChild(questionBubbles);
    // Scroll to the bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}


// Fetch and populate data
document.addEventListener('DOMContentLoaded', function() {
    // Get the current URL's query parameters
    const queryParams = new URLSearchParams(window.location.search);
    
    // Convert the query parameters to an object
    const paramsObject = {};
    queryParams.forEach((value, key) => {
        paramsObject[key] = value;
    });

    addLoadingBar("Loading Insight Cards & Questions")
    fetch('/main_first_insight', {
        method: 'POST', // Use POST to send data
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(paramsObject)
    })
        .then(response => response.json())
        .then(data => {
            addMessage(`Loaded ${data.insight_id} Insight Cards. Ask me anything.`);
            // Populate title
            const title = document.getElementById('title');
            title.innerHTML = `${data.title}`;

            const metaData = document.getElementById('meta-data');
            metaData.innerHTML = `${data.meta}`;
            
            // Populate Insights 
            const insightBlobs = document.getElementById('insight-blobs');
            insightBlobs.innerHTML = `${data.insight}`;
            
            // updateCount()

                //     addLoadingBar("Loading Initial Questions")
            
            addQuestionBubbles(data.questions)
            removeLoadingBar()

//     // Add initial question bubbles
//     addInitialQuestionBubbles();
    
//     // Scroll to the bottom of the chat pane
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            
        });

    // Get the dataset from URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const dataset = urlParams.get('dataset');
    
    // Fetch and populate the table
    fetch(`/get_dataframe?dataset=${dataset}`)
        .then(response => response.json())
        .then(data => {
            const tableFrame = document.getElementById('table-frame');
            const tabsMain = document.querySelector('.tabs-main');
            
            // Set table frame width to match tabs-main
            if (tabsMain) {
                tableFrame.style.width = tabsMain.offsetWidth + 'px';
            }
            
            // Create table element
            const table = document.createElement('table');
            table.className = 'display nowrap';
            
            // Create header with all columns
            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');
            
            // Ensure all columns are included
            const expectedColumns = [
                'employee_id', 'name', 'age', 'department', 'job_level',
                'location', 'salary', 'years_with_company', 'last_performance_rating',
                'is_remote', 'promotion_last_2_years', 'left_company', 'gender'
            ];
            
            expectedColumns.forEach(column => {
                const th = document.createElement('th');
                // Format the header text properly
                th.textContent = column
                    .split('_')
                    .map(word => word.toUpperCase())
                    .join(' ');
                
                // Add specific classes for width control
                th.className = column.replace(/_/g, '-');
                
                // Set minimum width for specific columns
                if (column === 'years_with_company' || column === 'last_performance_rating') {
                    th.style.minWidth = column === 'years_with_company' ? '180px' : '200px';
                }
                
                headerRow.appendChild(th);
            });
            
            thead.appendChild(headerRow);
            table.appendChild(thead);
            
            // Create body ensuring all columns are populated
            const tbody = document.createElement('tbody');
            data.data.forEach(row => {
                const tr = document.createElement('tr');
                // Map the data to match expected columns
                expectedColumns.forEach((column, index) => {
                    const td = document.createElement('td');
                    td.textContent = row[index] !== undefined ? row[index] : '';
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });
            table.appendChild(tbody);
            
            // Clear and add table to frame
            tableFrame.innerHTML = '';
            tableFrame.appendChild(table);
            // No DataTables initialization here!
        })
        .catch(error => console.error('Error loading table:', error));
});

function send_bubble(question) {
    populateInput(question)
    send()
}

function send() {
    removeBubbles()
    // Get the text from chatbotInput
    // logChatbotInput()
    const chatbotInput = document.getElementById('userInput')
    const userInputText = chatbotInput.value; // Save the user input text

    // Get the current URL and extract parameters
    var urlParams = new URLSearchParams(window.location.search);
    var params = {};
    for (const [key, value] of urlParams.entries()) {
        params[key] = value;
    }

    // Prepare data to send
    var data = {
        text: chatbotInput.value,
        ...params
    };
    add_user_message(`${userInputText}`)
     
    addLoadingBar(`Answering the question above (30-60 seconds)`)
    chatbotInput.value = '';
    handleSendButtonClick();
    const messagesContainer = document.getElementById('chatbotMessages');
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    // Send data to Flask using Fetch API
    fetch('/main_add_insight', {
        method: 'POST', // Use POST to send data
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
        .then(response => response.json())
        .then(data => {
            if (!data.isValid) {
                removeLoadingBar()
                addMessage(`Invalid: ${data.explanation}`);
                addQuestionBubbles(data.questions)
                
                
                return; // Exit the function if data is not valid
            }
            // Populate Insights 
            const insightBlobs = document.getElementById('insight-blobs');
            insightBlobs.innerHTML = `${data.insight} ${insightBlobs.innerHTML}`;
            
            addMessage(`Insight Cards (${data.insight_id}) Added to Dashboard`);
            // // Populate Meta Data
            // const metaData = document.getElementById('meta-data');
            // metaData.innerHTML = `${data.meta}`;

            // // Populate Summary
            // const summaryData = document.getElementById('summary-data');
            // summaryData.innerHTML = `${data.summary}`;

            // // Populate Action
            // const actionData = document.getElementById('action-data');
            // actionData.innerHTML = `${data.action}`;
            addQuestionBubbles(data.questions)
            removeLoadingBar()
            
        });
    }

document.getElementById('sendButton').addEventListener('click', send);

function updateCount() {
     const numInsightBlobs = countInsightBlobs();
        // alert(numInsightBlobs);
        const count = document.getElementById('count');
        count.textContent = numInsightBlobs;
}

// Function to add a single message
function addMessage( messageText) {
    const iconPath = '/static/assets/agent.webp'
    const container = document.getElementById('chatbotMessages');
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', 'agent-message');

    // Add agent icon
    const agentIcon = document.createElement('img');
    agentIcon.src = iconPath;
    agentIcon.alt = 'Agent Icon';
    messageElement.appendChild(agentIcon);

    // Add agent message text
    const messageContent = document.createElement('p');
    messageContent.textContent = messageText;
    messageElement.appendChild(messageContent);

    // Append the message to the chat pane
    container.appendChild(messageElement);
}

function add_user_message(message) {
    // Append user message to chat
    const messagesContainer = document.getElementById('chatbotMessages');
    const userMessageElement = document.createElement('div');
    userMessageElement.classList.add('chat-message', 'user-message');

    // Add user message text
    const userMessageText = document.createElement('p');
    userMessageText.textContent = message;
    userMessageElement.appendChild(userMessageText);

    // Add user icon
    const userIcon = document.createElement('img');
    userIcon.src = '/static/assets/user.png'; // replace with your icon path
    userIcon.alt = 'User Icon';
    userMessageElement.appendChild(userIcon);

    // Append user message to the chat pane
    messagesContainer.appendChild(userMessageElement);
}

function handleSendButtonClick() {
    const inputField = document.getElementById('userInput');
    const message = inputField.value.trim();
    if (message) {
        // Append user message to chat
        const messagesContainer = document.getElementById('chatbotMessages');
        const userMessageElement = document.createElement('div');
        userMessageElement.classList.add('chat-message', 'user-message');

        // Add user message text
        const userMessageText = document.createElement('p');
        userMessageText.textContent = message;
        userMessageElement.appendChild(userMessageText);

        // Add user icon
        const userIcon = document.createElement('img');
        userIcon.src = '/static/assets/user.png'; // replace with your icon path
        userIcon.alt = 'User Icon';
        userMessageElement.appendChild(userIcon);

        // Append user message to the chat pane
        messagesContainer.appendChild(userMessageElement);

        // Clear input field
        inputField.value = '';

        // Fetch new insight blob and update other blobs
        // fetchNewInsightBlob(message);
        updateBlobs();

        // Simulate bot response
        setTimeout(() => {
            const iconPath = '/static/assets/agent.webp';
            // Get the number of insight blobs
            
            addBotResponse(messagesContainer, "Getting Insight...", iconPath);

            addBotResponse(messagesContainer, `Added insight to the Dashboard`, iconPath);

            

            // Scroll to the bottom of the chat pane
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }, 1000);

       
        // Scroll to the bottom of the chat pane
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

function getQueryParam(name) {
            const urlParams = new URLSearchParams(window.location.search);
            return urlParams.get(name);
        }

        function updateGreeting() {
            const user = getQueryParam('user');
            const greetingElement = document.getElementById('greeting');
            if (user) {
                greetingElement.textContent = `Greetings, ${user}!`;
            } else {
                greetingElement.textContent = 'Hello! Chat with Poirot';
            }
        }

        // Call the function to update the greeting when the page loads
        updateGreeting();