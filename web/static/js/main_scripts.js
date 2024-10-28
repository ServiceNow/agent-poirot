// Function to fetch questions from the server
function fetchQuestions(callback) {
    const insights = document.getElementById('insight-blobs');

    fetch('/questions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ "insights": insights.textContent }),
    })
    .then(response => response.json())
    .then(data => {
        callback(data);
    })
    .catch(error => console.error('Error sending text to server:', error));
}

// Function to populate the input field with a question
function populateInput(question) {
    const inputField = document.getElementById('userInput');
    inputField.value = question;
    // Adjust the height of the textarea
    inputField.style.height = 'auto'; // Reset height to auto to calculate the new height
    // inputField.style.height = `${inputField.scrollHeight}px`; // Set height to scrollHeight
}

// Function to add initial question bubbles
function addInitialQuestionBubbles() {
    fetchQuestions(questions => {
        const messagesContainer = document.getElementById('chatbotMessages');
        const initialQuestionBubbles = document.createElement('div');
        initialQuestionBubbles.classList.add('recommended-questions');
        questions.forEach(question => {
            const questionBubble = document.createElement('div');
            questionBubble.classList.add('question-oval');
            questionBubble.textContent = question;
            questionBubble.onclick = () => populateInput(question);
            initialQuestionBubbles.appendChild(questionBubble);
        });
        messagesContainer.appendChild(initialQuestionBubbles);
    });
    

}

// Function to add question bubbles after bot response
// function addQuestionBubbles() {
//     fetchQuestions(questions => {
//         const messagesContainer = document.getElementById('chatbotMessages');
//         const questionBubbles = document.createElement('div');
//         questionBubbles.classList.add('recommended-questions');
//         questions.forEach(question => {
//             const questionBubble = document.createElement('div');
//             questionBubble.classList.add('question-oval');
//             questionBubble.textContent = question;
//             questionBubble.onclick = () => populateInput(question);
//             questionBubbles.appendChild(questionBubble);
//         });
//         messagesContainer.appendChild(questionBubbles);
//         // Scroll to the bottom
//         messagesContainer.scrollTop = messagesContainer.scrollHeight;
//     });
// }

// Function to handle question click
function handleQuestionClick(question) {
    populateInput(question);
}

// Function to populate input field with question and remove bubbles
function populateInput(question) {
    const inputField = document.getElementById('userInput');
    inputField.value = question;
    inputField.focus();
    document.querySelector('.recommended-questions').remove();
}

// Function to fetch new insight-blob from Flask
// Function to fetch new insight-blob from Flask
function fetchNewInsightBlob(userMessage) {
    fetch('/new-insight', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage }),
    })
        .then(response => response.text())
        .then(data => {
            const insightBlobs = document.getElementById('insight-blobs');
            const newInsightBlob = document.createElement('div');
            newInsightBlob.classList.add('insight-blob');
            newInsightBlob.innerHTML = data;
            insightBlobs.appendChild(newInsightBlob);
        })
        .catch(error => console.error('Error fetching new insight blob:', error));
}


// Function to update meta, summary, and action blobs
function updateBlobs() {
    const metaData = document.getElementById('meta-data');
    const summaryData = document.getElementById('summary-data');
    const actionData = document.getElementById('action-data');

    // metaData.innerHTML = 'updated';
    // summaryData.innerHTML = 'updated';
    // actionData.innerHTML = 'updated';
}
// Send message function


function addBotResponse(messagesContainer, messageText, iconPath) {
    const botResponse = document.createElement('div');
    botResponse.classList.add('chat-message', 'agent-message');

    // Add agent icon
    const agentIcon = document.createElement('img');
    agentIcon.src = iconPath; // replace with your icon path
    agentIcon.alt = 'Agent Icon';
    botResponse.appendChild(agentIcon);
    
    // Add agent message text
    const agentMessageText = document.createElement('p');
    agentMessageText.textContent = messageText;
    botResponse.appendChild(agentMessageText);

    // Append agent message to the chat pane
    messagesContainer.appendChild(botResponse);
}





// Function to toggle view more/less
function toggleViewMore(link) {
    const details = link.nextElementSibling;
    if (details.style.display === 'none' || details.style.display === '') {
        details.style.display = 'block';
        link.textContent = 'View Less';
    } else {
        details.style.display = 'none';
        link.textContent = 'View More';
    }
}






// Function to handle image popup
document.addEventListener('DOMContentLoaded', function() {
    const insightImages = document.querySelectorAll('.insight-blob a img');

    insightImages.forEach(image => {
        image.addEventListener('click', function(event) {
            event.preventDefault();
            const imageUrl = this.parentElement.href;

            // Get the dimensions of the browser window
            const width = 50;
            const height = 37.5;
            const left = (screen.width / 2) - (width / 2);
            const top = (screen.height / 2) - (height / 2);

            // Open the pop-up window at the center of the screen
            const popupWindow = window.open('', 'Image Popup', `width=${width}rem,height=${height}rem,top=${top}px,left=${left}px`);
            popupWindow.document.write(`
                <html>
                <head>
                    <title>Image Popup</title>
                    <style>
                        body {
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            height: 100vh;
                            margin: 0;
                        }
                        img {
                            max-width: 100%;
                            max-height: 100%;
                        }
                    </style>
                </head>
                <body>
                    <img src="${imageUrl}" alt="Plot Image">
                </body>
                </html>
            `);
        });
    });
});

// Function to initialize DataTable
function initializeDataTable() {
    $('#data-table').DataTable({
        scrollY: '400px',
        scrollCollapse: true,
        paging: false,
        searching: false, // Disable search box
        info: false // Disable info display
    });
}

// Function to fetch DataFrame data and populate the table and stats
async function fetchDataFrameData() {
    try {
        // Get the current URL's query parameters
        const queryParams = new URLSearchParams(window.location.search);
        
        // Convert the query parameters to an object
        const paramsObject = {};
        queryParams.forEach((value, key) => {
            paramsObject[key] = value;
        });


        const response = await fetch('/get_dataframe', {
            method: 'POST', // Use POST to send data
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(paramsObject)
        })

        const data = await response.json();

        populateTable(data.table);
        // Optional: Populate stats if needed
        // populateStats(data.stats);

        // Initialize DataTable after table is populated
        setTimeout(initializeDataTable, 0);
    } catch (error) {
        console.error('Error fetching data:', error);
    }
}

function countInsightBlobs() {
    const questionsSection = document.getElementById('questions-section');
    if (questionsSection) {
        // Count the number of elements with the class 'insight-blob'
        const insightBlobs = questionsSection.getElementsByClassName('insight-blob-eval');
        return insightBlobs.length;
    }
    return 0;
}

// Function to populate the table
function populateTable(tableData) {
    let tableFrame = document.getElementById('table-frame');
    let table = '<table id="data-table" class="display"><thead><tr>';

    // Add headers
    for (let header of tableData.headers) {
        table += `<th>${header}</th>`;
    }
    table += '</tr></thead><tbody>';

    // Add rows
    for (let row of tableData.rows) {
        table += '<tr>';
        for (let cell of row) {
            table += `<td>${cell}</td>`;
        }
        table += '</tr>';
    }
    table += '</tbody></table>';

    tableFrame.innerHTML = table;
}

// Optional: Function to populate stats (uncomment if needed)
// function populateStats(statsData) {
//     let statsBlob = document.getElementById('stats-blob');
//     let stats = '<ul>';
//     for (let key in statsData) {
//         stats += `<li>${key}: ${statsData[key]}</li>`;
//     }
//     stats += '</ul>';
//     statsBlob.innerHTML = stats;
// }

// Function to fetch DataFrame data and populate the table and stats for /get_dataframe2
async function fetchDataFrameData2() {
    try {
        const response = await fetch('/get_dataframe2');
        const data = await response.json();

        // Populate the table using populateTable2 function
        populateTable2(data.table, data.stats);

        // Initialize DataTable after table is populated
        setTimeout(initializeDataTableStats, 0);
    } catch (error) {
        console.error('Error fetching data from /get_dataframe2:', error);
    }
}

// Function to populate table and stats for /get_dataframe2
function populateTable2(tableData, statsData) {
    let tableFrame = document.getElementById('table-frame-stats');
    let table = '<table id="data-table-stats" class="display"><thead><tr>';

    // Add headers
    for (let header of tableData.headers) {
        table += `<th>${header}</th>`;
    }
    table += '</tr></thead><tbody>';

    // Add rows
    for (let row of tableData.rows) {
        table += '<tr>';
        for (let cell of row) {
            table += `<td>${cell}</td>`;
        }
        table += '</tr>';
    }
    table += '</tbody></table>';

    tableFrame.innerHTML = table;

    // Populate stats
    let statsBlob = document.getElementById('stats-blob');
    let stats = '<ul>';
    for (let key in statsData) {
        stats += `<li>${key}: ${statsData[key]}</li>`;
    }
    stats += '</ul>';
    statsBlob.innerHTML = stats;
}

// Function to initialize DataTable for stats
function initializeDataTableStats() {
    $('#data-table-stats').DataTable({
        scrollY: '400px',
        scrollCollapse: true,
        paging: false,
        searching: false, // Disable search box
        info: false // Disable info display
    });
}

// Set the default tab to open
document.addEventListener('DOMContentLoaded', function() {
    document.querySelector('.tablink').click();
});

// Event listener for the 'Dataset' tab button
document.querySelector('button[onclick="openTab(event, \'Dataset\')"]').addEventListener('click', fetchDataFrameData);

// TODO: confirm that this function is necessary. It appears to look for an element that doesn't exist.
document.querySelector('button[onclick="openTab(event, \'DatasetSummary\')"]').addEventListener('click', fetchDataFrameData2);
