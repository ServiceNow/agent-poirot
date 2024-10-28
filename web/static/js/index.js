function showTab(tabName) {
    console.log({tabName})
    const tabContents = document.querySelectorAll('.tabcontent');
    const tabLinks = document.querySelectorAll('.tablink');
    
    tabContents.forEach((tab) => {
        tab.classList.remove('active');
    });

    tabLinks.forEach((tab) => {
        tab.classList.remove('active');
    });

    document.getElementById(tabName).classList.add('active');
    document.querySelector(`.tablink[onclick="showTab('${tabName}')"]`).classList.add('active');
}

function toggleSelection(element) {
    element.classList.toggle('selected');
}

/* ANALYZE SELECTED DATASET */
/* ------------------------- */
function analyzeSelectedDatasets() {
    const selectedElements = document.querySelectorAll('.blob.selected');
    const selectedTopics = Array.from(selectedElements).map(el => el.getAttribute('data-topic'));
    const user = document.getElementById('email-input').value;
    // Check if email is not empty
    if (user.trim() === '') {
        alert('Please enter your email.');
        return;
    }

    // Implement your analysis logic here
    console.log(`Analyzing datasets: ${selectedTopics.join(', ')}`);
    const dataset = selectedTopics.join(',')
    if (dataset.trim() === '') {
        alert('Please select a dataset.');
        return;
    }
    // Example redirect:
    const queryString = new URLSearchParams({
        dataset: dataset, // Join selected topics with comma
        user: user,
    }).toString();

    const url = `/main?${queryString}`;
    window.location.href = url;
}

function analyzeData() {
    const url = document.querySelector('.search-bar input').value;
    // Implement data analysis logic
    console.log(`Analyzing data from URL: ${url}`);
}

function evals() {
    // Get the email value from the input field
    const email = document.getElementById('email-input').value;

    // Check if email is not empty
    if (email.trim() === '') {
        alert('Please enter your email.');
        return;
    }

    // Perform an HTTP POST request to the /eval endpoint
    fetch('/eval', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email: email }),
    })
    .then(response => response.text())
    .then(data => {
        // Handle the response from the server (optional)
        console.log('Success:', data);
            // Example redirect:
    const queryString = new URLSearchParams({
        dataset: "csm", // Join selected topics with comma
        user: "guest",
    }).toString();

    const url = `/eval_comparison?${queryString}`;
    window.location.href = url;
    })
    .catch((error) => {
        // Handle any errors (optional)
        console.error('Error:', error);
        alert('An error occurred while submitting your evaluation.');
    });
}


function handleLogin() {
    // alert("erwre");
    // const selectedElements = document.querySelectorAll('.blob.selected');
    // const selectedTopics = Array.from(selectedElements).map(el => el.getAttribute('data-topic'));
    const user = document.getElementById('email-input').value;
    // Check if email is not empty
    if (user.trim() === '') {
        alert('Please enter your email.');
        return;
    }


    // Implement your analysis logic here
    // console.log(`Analyzing datasets: ${selectedTopics.join(', ')}`);
    const dataset = document.getElementById('dataset-input').value;
    if (dataset.trim() === '') {
        alert('Please select a dataset.');
        return;
    }
    // Example redirect:
    const queryString = new URLSearchParams({
        dataset: dataset, // Join selected topics with comma
        user: user,
    }).toString();

    const url = `/main?${queryString}`;
    window.location.href = url;

    // const emailInput = document.getElementById("email-input");
    // const datasetInput = document.getElementById("dataset-input");
    // const tabs = document.getElementById('tabs');
    // const exampleDatasets = document.getElementById('example-datasets');
    // const loginButton = document.getElementById('login-button');

    // if (emailInput.value.trim() === '') {
    //     alert('Please enter your email.');
    //     return;
    // }
    // if (datasetInput.value.trim() === '') {
    //     alert('Please enter a dataset.');
    //     return;
    // }

    // // Get the email value from the input field
    // const email = document.getElementById('email-input').value;

    // // Check if email is not empty
    // if (email.trim() === '') {
    //     alert('Please enter your email.');
    //     return;
    // }

    // // Perform an HTTP POST request to the /eval endpoint
    // fetch('/eval', {
    //     method: 'POST',
    //     headers: {
    //         'Content-Type': 'application/json',
    //     },
    //     body: JSON.stringify({ email: email }),
    // })
    // .then(response => response.text())
    // .then(data => {
    //     // Handle the response from the server (optional)
    //     console.log('Success:', data);
    //         // Example redirect:
    // const queryString = new URLSearchParams({
    //     dataset: "csm", // Join selected topics with comma
    //     user: "guest",
    // }).toString();

    // const url = `/eval_comparison?${queryString}`;
    // window.location.href = url;
    // })
    // .catch((error) => {
    //     // Handle any errors (optional)
    //     console.error('Error:', error);
    //     alert('An error occurred while submitting your evaluation.');
    // });

    // if (tabs.classList.contains('hidden')) {
    //     // If hidden, show the elements and change the button text
    //     tabs.classList.remove('hidden');
    //     exampleDatasets.classList.remove('hidden');
    //     exampleDatasets.classList.add('active');
    //     loginButton.textContent = "Login Successful (Click to Edit)";
    //     emailInput.disabled = true;
    // } else {
    //     // If visible, hide the elements and revert the button text
    //     tabs.classList.add('hidden');
    //     exampleDatasets.classList.add('hidden');
    //     exampleDatasets.classList.remove('active');
    //     loginButton.textContent = "Login";
    //     emailInput.disabled = false;
    // }
}