<!-- Drop this inside <div id="ui-panel"> right below the leaderboard -->
<div id="voting-booth" style="background: #e94560; padding: 15px; border-radius: 10px; margin-bottom: 15px; text-align: center; box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);">
    <h2 class="glowing-text" style="margin-top: 0;">👠 The Runway</h2>
    <p>Judging: <b id="target-model" style="font-size: 1.2em;">Nobody</b></p>
    <p>Wearing: <i id="target-outfit">Nothing</i></p>
    
    <input type="range" id="score-slider" min="1" max="10" value="5" 
           oninput="document.getElementById('score-display').innerText = this.value"
           style="width: 100%; margin: 10px 0;">
           
    <div style="font-size: 1.5em; font-weight: bold; margin-bottom: 10px;">
        <span id="score-display">5</span>/10
    </div>
    
    <button onclick="submitVote()" style="background: #1a1a2e; width: 45%;">Slay/Nay? 💅</button>
    <button onclick="nextModel()" style="background: #0f3460; width: 45%;">Skip ⏭️</button>
    
    <div id="vote-feedback" style="margin-top: 10px; font-weight: bold; height: 20px;"></div>
</div>
