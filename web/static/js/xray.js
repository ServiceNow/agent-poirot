let currentButton = null;
const submitFeedbackButton = document.getElementById('feedbackButton');
const toggleButtons = Array.from(document.getElementsByClassName('toggler'));

const choiceState = {
    choice: null
}

function resetButtons() {
    toggleButtons.forEach(button => {
        button.classList.remove('active');
    });
    submitFeedbackButton.disabled = true;
}


const compareViewport = document.getElementById('compare-viewport');
function startFetchUx() {
    console.log('startFetchUx');
    handleCardHighlight(-1);
    console.log(compareViewport);
    compareViewport.style.display = 'hidden';
}

function endFetchUx() {
    console.log('endFetchUx');
    compareViewport.style.display = 'flex';
}

function fetchDataEvalArcade() {
    startFetchUx();
   fetch('/xray_load_insight_card', {
        method: 'POST', // Use POST to send data
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({"dataset":"csm"})
    })
        .then(response => response.json())
        .then(data => {
            count = data.count;
            user = data.username;

            if (data.n_insights === 0) {
                // TODO: better UX here.
                alert("You have evaluated all insights. Thank you for your feedback!");
            } else {
                // Populate title
                // const title = document.getElementById('title');
                // title.innerHTML = `${data.title}`;

                // Populate Insights
                const insightBlob1 = document.getElementById('insight-blob1');
                insightBlob1.innerHTML = `${data.insight_card_a}`;

                const insightBlob2 = document.getElementById('insight-blob2');
                insightBlob2.innerHTML = `${data.prompt}`;

                // const banner = document.getElementById('banner-text');
                // banner.innerHTML = `Dataset: ${data.dataset}`
                
                // const timestamp = document.getElementById('timestamp');
                // timestamp.innerHTML = `${data.timestamp}`

                // const countElement = document.getElementById('count');
                // countElement.innerHTML = count;
            }
        })
        .then(resetButtons)
        .then(endFetchUx)
        .catch(error => {
            console.error('Error fetching data:', error);
        });
}


const handleCardHighlight = (id) => {
    const selectedClass = 'selected';
    const insightCardElementIds = ['insight-blob1', 'insight-blob2'];
    
    const insightCardElements = insightCardElementIds.map(id => document.getElementById(id)) 

    const selectetInsightCardIds = id === 99 ? insightCardElementIds : id >= 0 ? [insightCardElementIds[id]] : [];
    
    insightCardElements.forEach(element => {
        if (selectetInsightCardIds.includes(element.id)) {
            element.classList.add(selectedClass);
        }
        else {
            element.classList.remove(selectedClass);
        }
    })
}

function runPrompt(evt, tabName) {
   fetch('/xray_run_prompt', {
        method: 'POST', // Use POST to send data
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({"dataset":"csm"})
    })
        .then(response => response.json())
        .then(data => {
            count = data.count;
            user = data.username;

            if (data.n_insights === 0) {
                // TODO: better UX here.
                alert("You have evaluated all insights. Thank you for your feedback!");
            } else {
                // Populate title
                // const title = document.getElementById('title');
                // title.innerHTML = `${data.title}`;

                // Populate Insights
                const insightBlob1 = document.getElementById('insight-blob1');
                insightBlob1.innerHTML = `${data.insight_card_a}`;

                const insightBlob2 = document.getElementById('insight-blob2');
                insightBlob2.innerHTML = `${data.prompt}`;

                // const banner = document.getElementById('banner-text');
                // banner.innerHTML = `Dataset: ${data.dataset}`
                
                // const timestamp = document.getElementById('timestamp');
                // timestamp.innerHTML = `${data.timestamp}`

                // const countElement = document.getElementById('count');
                // countElement.innerHTML = count;
            }
        })
        .then(resetButtons)
        .then(endFetchUx)
        .catch(error => {
            console.error('Error fetching data:', error);
        });
}

document.addEventListener('DOMContentLoaded', () => {
    fetchDataEvalArcade();
});