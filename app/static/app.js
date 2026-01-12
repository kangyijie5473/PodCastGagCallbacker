let currentIndexId = null
let currentAudioUrl = null

const fileInput = document.getElementById('file')
const btnIndex = document.getElementById('btnIndex')
const btnSearch = document.getElementById('btnSearch')
const qInput = document.getElementById('q')
const resultsDiv = document.getElementById('results')
const player = document.getElementById('player')

btnIndex.onclick = async () => {
  const f = fileInput.files[0]
  if (!f) return
  const fd = new FormData()
  fd.append('file', f)
  const r = await fetch('/api/index', { method: 'POST', body: fd })
  const j = await r.json()
  currentIndexId = j.index.split('/').pop()
  currentAudioUrl = j.audio_url
  player.src = currentAudioUrl
}

btnSearch.onclick = async () => {
  const q = qInput.value.trim()
  if (!q) return
  const params = new URLSearchParams({ q, id: currentIndexId || '' })
  const r = await fetch('/api/search?' + params.toString())
  const arr = await r.json()
  resultsDiv.innerHTML = ''
  arr.forEach(x => {
    const div = document.createElement('div')
    div.className = 'card'
    div.innerHTML = `<div>${escapeHtml(x.text)}</div><div class="time">${fmtTime(x.start)} - ${fmtTime(x.end)} | 说话人 ${x.speaker} | 分数 ${x.score.toFixed(3)}${x.laughter>0? ' | 笑声':''}</div><div><button>播放</button></div>`
    div.querySelector('button').onclick = () => {
      if (player.src !== currentAudioUrl && currentAudioUrl) player.src = currentAudioUrl
      player.currentTime = x.start
      player.play()
    }
    resultsDiv.appendChild(div)
  })
}

function fmtTime(t){
  const s = Math.floor(t)
  const m = Math.floor(s/60)
  const ss = s%60
  return `${m}:${ss.toString().padStart(2,'0')}`
}

function escapeHtml(str){
  return str.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))
}
