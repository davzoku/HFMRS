const API = 'http://localhost:8000/recommend';
const USER_SETTINGS = ['method', 'recommendNo', 'sortBy']
const DEFAULT_METHOD = "cosine"
const DEFAULT_RECOMMEND_NO = 5;
const DEFAULT_SORT_BY = 'none';

const modelSideSection = document.querySelector('.pt-8.border-gray-100.md\\:col-span-5.pt-6.md\\:pb-24.md\\:pl-6.md\\:border-l.order-first.md\\:order-none');

// logo assets
const img = document.createElement('img');
img.src = chrome.runtime.getURL('images/hfmrs-18.svg');
const svgWrapper = document.createElement('span');
svgWrapper.classList.add('mr-1', 'inline', 'self-center', 'flex-none');
svgWrapper.innerHTML = img.outerHTML;
const textWrapper = document.createElement('span');
textWrapper.textContent = ' Similar Models';

let hfmrsDiv = document.createElement('div')

let url = document.querySelector("[property='og:url']").getAttribute("content");
let modelId;
let hfmrsResultElement = "hfmrs-result"

const testResponse = [
    { "modelId": "albert-base-v1", "score": 0.989 },
    { "modelId": "albert-base-v2", "score": 0.890 }
  ];

function stripHuggingFaceUrl(url) {
    const prefix = "https://huggingface.co/";
    if (url.startsWith(prefix)) {
      return url.slice(prefix.length);
    }
    return url;
  }

function setupUI() {
    let template = `<div id="hfmrs-title"><h2 id="hfmrs-title-h2" class="mb-5 flex items-baseline overflow-hidden whitespace-nowrap text-smd font-semibold text-gray-800"></h2></div><div id="hfmrs-result">Searching...</div>`
    hfmrsDiv.innerHTML = template
    hfmrsDiv.className = 'hfmrs'

    const divider = document.createElement('div')
    divider.classList.add('divider-column-vertical')   

    modelSideSection.appendChild(divider);
    modelSideSection.appendChild(hfmrsDiv);

    const titleH2 = document.getElementById('hfmrs-title-h2');
    titleH2.appendChild(svgWrapper);
    titleH2.appendChild(textWrapper);
}

function getSettings(callback) {
  chrome.storage.sync.get(USER_SETTINGS, function (data) {
      var settings = {
          method: data.method || DEFAULT_METHOD,
          recommendNo: data.recommendNo || DEFAULT_RECOMMEND_NO,
          sortBy: data.sortBy || DEFAULT_SORT_BY
      };
      callback(settings);
  });
}

function search(modelId) {
    getSettings(function (settings) {
      var recommendNo = settings.recommendNo;
      var method = settings.method;
      var sortBy = settings.sortBy;
      fetch(`${API}?model_id=${encodeURIComponent(modelId)}&method=${encodeURIComponent(method)}&recommend_no=${encodeURIComponent(recommendNo)}&sort_by=${encodeURIComponent(sortBy)}`)
      .then(response => {
          response.json().then(res => {
              resultDiv = document.getElementById(hfmrsResultElement);
              displayModelScoresTable(res, resultDiv)
          })
      })
  });
}

function displayModelScoresList(models, resultDiv) {
    const list = document.createElement('ul');
    for (let i = 0; i < models.length; i++) {
      const item = document.createElement('li');
      const model = models[i];
      // show score to 5 decimal place
      const text = document.createTextNode(`${model.model_id}: x ${model.score.toPrecision(5)}`);
      item.appendChild(text);
      list.appendChild(item);
    }
    resultDiv.innerHTML = '';
    resultDiv.appendChild(list);
  }
  
  function displayModelScoresTable(models, resultDiv) {
    const table = document.createElement('table');
    table.style.cssText = `
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        border: none;
    `;
    
    const header = table.createTHead();
    const row = header.insertRow();
    const modelIdHeader = row.insertCell();
    const scoreHeader = row.insertCell();
    modelIdHeader.innerHTML = '<b>Model Id</b>';
    scoreHeader.innerHTML = '<b>Similarity Score</b>';
    
    const body = table.createTBody();
    for (let i = 0; i < models.length; i++) {
        const row = body.insertRow();
        const model = models[i];
        const modelIdCell = row.insertCell();
        const scoreCell = row.insertCell();
        const modelIdLink = document.createElement('a');
        modelIdLink.href = `https://huggingface.co/${model.model_id}`;
        modelIdLink.textContent = model.model_id;
        modelIdLink.style.textDecoration = 'underline';
        modelIdCell.appendChild(modelIdLink);
        // show score to 5 decimal place
        scoreCell.innerText = model.score.toPrecision(5);
    }
    
    resultDiv.innerHTML = '';
    resultDiv.appendChild(table);
    
    table.querySelectorAll('th, td').forEach(cell => {
        cell.style.cssText = `
            padding: 8px;
            text-align: left;
            border: none;
        `;
    });
}


modelId = stripHuggingFaceUrl(url);
setupUI();
search(modelId)

