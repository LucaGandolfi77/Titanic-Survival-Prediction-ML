let currentTarget = null;

async function nextModel() {
    const res = await fetch('/api/strut');
    const data = await res.json();
    
    if (data.username) {
        currentTarget = data.username;
        document.getElementById('target-model').innerText = data.username;
        document.getElementById('target-outfit').innerText = data.outfit;
        document.getElementById('vote-feedback').innerText = "";
        
        // Visual flair: reset player rotation and change background slightly
        playerMesh.rotation.y = 0; 
        scene.background = new THREE.Color(0xffb6c1);
    } else {
        document.getElementById('target-model').innerText = "Ghost Town";
    }
}

async function submitVote() {
    if (!currentTarget) return;
    const score = parseInt(document.getElementById('score-slider').value);

    const res = await fetch('/api/vote', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ target_username: currentTarget, score: score })
    });
    const data = await res.json();

    // Visual and text feedback based on how harsh the judge is
    const feedback = document.getElementById('vote-feedback');
    if (score >= 8) {
        feedback.innerText = "🔥 ABSOLUTE FIRE!";
        feedback.style.color = "#ffdd59";
        scene.background = new THREE.Color(0xffd700); // Gold flash in 3D scene!
        playerMesh.scale.set(1.2, 1.2, 1.2); // Player swells with pride
    } else if (score <= 3) {
        feedback.innerText = "🗑️ TRASH!";
        feedback.style.color = "#1a1a2e";
        scene.background = new THREE.Color(0x333333); // Dark gloomy flash
        playerMesh.scale.set(0.8, 0.8, 0.8); // Player shrinks in shame
    } else {
        feedback.innerText = "Meh. Play it safe.";
        feedback.style.color = "#fff";
        playerMesh.scale.set(1, 1, 1);
    }

    // Automatically load the next victim after 2 seconds
    setTimeout(nextModel, 2000);
}

// Call this once at the bottom of your script to load the first player
nextModel();
