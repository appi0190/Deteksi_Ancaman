function search() {
    let q = document.getElementById("query").value.trim();
    if (q === "") return;

    fetch(`/search?q=` + encodeURIComponent(q))
        .then(res => res.json())
        .then(data => {
            let container = document.getElementById("results");
            container.innerHTML = "";

            data.forEach(item => {
                container.innerHTML += `
                    <div class="result-card">

                        <div class="label">${item.method} Match — Score: ${item.score.toFixed(3)}</div>

                        <div class="result-title">${item.attack_type} (${item.platform})</div>

                        <div class="value"><span class="label">Date:</span> ${item.date}</div>
                        <div class="value"><span class="label">Impact:</span> ${item.impact}</div>
                        <div class="value"><span class="label">Description:</span> ${item.description}</div>
                        <div class="value"><span class="label">Mitigation:</span> ${item.mitigation}</div>

                    </div>
                `;
            });
        });
}
