function loadPage(pageName) {
    const mainContent = document.getElementById('main-content');
    fetch(pageName + '.html')
        .then(response => response.text())
        .then(html => {
            mainContent.innerHTML = html;
        })
        .catch(error => console.error('Error loading page: ', error));
}

function calculateIndex() {
    const place = document.getElementById('place').value;
    const year = document.getElementById('year').value;
    // Calculate water quality index based on place and year
    const result = "Water quality index for " + place + " in " + year + " is calculated...";
    document.getElementById('result').innerText = result;
}