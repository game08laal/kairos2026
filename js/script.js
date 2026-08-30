// Link da IA gerado pelo Teachable Machine
const URL_MODELO = "https://teachablemachine.withgoogle.com/models/5ac2tL3-X/";

let streams = {};
let cameras = [];
let model, maxPredictions;

// Função auxiliar para atualizar o visual do status
function atualizarStatus(numero, tipo, texto) {
    const elStatus = document.getElementById(`status${numero}`);
    if (elStatus) {
        elStatus.className = `status ${tipo}`; // online, offline, inicializando
        elStatus.innerText = texto;
    }
}

// Função auxiliar para pegar a hora atual formatada (HH:MM:SS)
function obterHoraAtual() {
    const agora = new Date();
    return agora.toTimeString().split(' ')[0];
}

// Inicia as câmeras assim que a página abre
async function listarEComecarCameras() {
    try {
        const streamTeste = await navigator.mediaDevices.getUserMedia({ video: true });
        streamTeste.getTracks().forEach(track => track.stop());

        const dispositivos = await navigator.mediaDevices.enumerateDevices();
        const camerasEncontradas = dispositivos.filter(device => device.kind === "videoinput");
        
        // Evita duplicidade de IDs devido ao HUB USB
        cameras = camerasEncontradas.filter((cam, index, self) =>
            index === self.findIndex((c) => c.deviceId === cam.deviceId)
        );

        console.log("Câmeras físicas encontradas:", cameras.length);

        // Marca como offline blocos sem câmera física conectada
        for (let i = 1; i <= 4; i++) {
            if (i > cameras.length) {
                atualizarStatus(i, 'offline', 'Offline');
                document.getElementById(`captura${i}`).innerText = "Última captura: --:--:--";
            }
        }

        // Liga as câmeras com uma pequena pausa de segurança entre elas
        for (let i = 0; i < cameras.length; i++) {
            const numeroBloco = i + 1;
            if (numeroBloco <= 4) {
                await ligarCameraPorIndice(numeroBloco, i);
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }

    } catch (erro) {
        alert("Erro ao iniciar o sistema: " + erro);
    }
}

async function ligarCameraPorIndice(numeroBloco, indiceCamera) {
    const video = document.getElementById(`camera${numeroBloco}`);
    if (!video || !cameras[indiceCamera]) return;

    try {
        atualizarStatus(numeroBloco, 'inicializando', 'Inicializando');

        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                deviceId: { exact: cameras[indiceCamera].deviceId },
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        });

        streams[numeroBloco] = stream;
        video.srcObject = stream;
        
        atualizarStatus(numeroBloco, 'online', 'Online');
        document.getElementById(`captura${numeroBloco}`).innerText = `Última captura: ${obterHoraAtual()}`;

    } catch (erro) {
        console.error(`Erro no bloco ${numeroBloco}:`, erro);
        atualizarStatus(numeroBloco, 'offline', 'Offline');
    }
}

// ==========================================
// INTEGRANDO A INTELIGÊNCIA ARTIFICIAL (IA)
// ==========================================

async function inicializarIA() {
    const modelURL = URL_MODELO + "model.json";
    const metadataURL = URL_MODELO + "metadata.json";

    try {
        console.log("Carregando Inteligência Artificial...");
        model = await tmImage.load(modelURL, metadataURL);
        maxPredictions = model.getTotalClasses();
        console.log("IA Carregada com Sucesso!");
        
        loopIA();
    } catch (e) {
        console.error("Erro ao carregar o modelo de IA:", e);
    }
}

// Criamos contadores para cada uma das 4 câmeras guardarem o histórico de estabilidade
let contadoresBacteria = { 1: 0, 2: 0, 3: 0, 4: 0 };
let contadoresVazio = { 1: 0, 2: 0, 3: 0, 4: 0 };
let estadoAtualDetectado = { 1: false, 2: false, 3: false, 4: false };

async function loopIA() {
    if (!model) return;

    for (let numeroBloco = 1; numeroBloco <= 4; numeroBloco++) {
        const videoElement = document.getElementById(`camera${numeroBloco}`);
        
        if (videoElement && videoElement.readyState === 4 && streams[numeroBloco]) {
            await prever(videoElement, numeroBloco);
        }
    }
    window.requestAnimationFrame(loopIA);
}

async function prever(video, numeroBloco) {
    const prediction = await model.predict(video);
    
    let veBactériaAgora = false;

    for (let i = 0; i < maxPredictions; i++) {
        const nomeClasse = prediction[i].className;
        const probabilidade = prediction[i].probability;

        if (nomeClasse === "Bactéria" && probabilidade > 0.90) {
            veBactériaAgora = true;
        }
    }

    // Lógica do Filtro de Estabilidade (Debounce)
    if (veBactériaAgora) {
        contadoresBacteria[numeroBloco]++;
        contadoresVazio[numeroBloco] = 0;

        if (contadoresBacteria[numeroBloco] >= 5) {
            estadoAtualDetectado[numeroBloco] = true;
        }
    } else {
        contadoresVazio[numeroBloco]++;
        contadoresBacteria[numeroBloco] = 0;

        if (contadoresVazio[numeroBloco] >= 5) {
            estadoAtualDetectado[numeroBloco] = false;
        }
    }

    const nomesCameras = {
        1: "Vista Superior",
        2: "Lateral Esquerda",
        3: "Lateral Direita",
        4: "Macro"
    };
    const nomeAtual = nomesCameras[numeroBloco] || `Câmera ${numeroBloco}`;

    if (estadoAtualDetectado[numeroBloco]) {
        atualizarStatus(numeroBloco, 'online status-bacteria', `${nomeAtual} - Bactéria!`);
        document.getElementById(`captura${numeroBloco}`).innerText = `Última captura: ${obterHoraAtual()}`;
    } else {
        atualizarStatus(numeroBloco, 'online', 'Online');
    }
}

// ==========================================
// CONTROLES DE CAPTURA E TELA CHEIA (TOGGLE)
// ==========================================

function capturarImagemManualmente(numeroBloco) {
    const video = document.getElementById(`camera${numeroBloco}`);
    const imgFoto = document.getElementById(`foto${numeroBloco}`);
    
    if (!video || !streams[numeroBloco]) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataURL = canvas.toDataURL('image/png');
    imgFoto.src = dataURL;
    imgFoto.style.display = 'block';

    document.getElementById(`captura${numeroBloco}`).innerText = `Última captura: ${obterHoraAtual()}`;
}

// Alterna entre expandir e restaurar o bloco (Estilo YouTube)
function alternarExpandirCamera(numeroBloco) {
    const bloco = document.getElementById(`bloco${numeroBloco}`);
    const btn = document.getElementById(`btn-expandir-${numeroBloco}`);
    
    if (!bloco || !btn) return;

    const jaEstaExpandido = bloco.classList.contains('expandido');

    if (jaEstaExpandido) {
        bloco.classList.remove('expandido');
        document.body.classList.remove('em-tela-cheia');
        btn.innerText = '⛶'; // Ícone de expandir
    } else {
        // Garante que nenhum outro bloco fique preso em modo expandido
        document.querySelectorAll('.bloco').forEach(b => b.classList.remove('expandido'));
        
        // Reseta o ícone de todos os outros botões para '⛶'
        for (let i = 1; i <= 4; i++) {
            const b = document.getElementById(`btn-expandir-${i}`);
            if (b) b.innerText = '⛶';
        }

        bloco.classList.add('expandido');
        document.body.classList.add('em-tela-cheia');
        btn.innerText = '🗗'; // Ícone de restaurar/reduzir
    }
}

// Atalho da tecla ESC para restaurar a tela e os ícones dos botões
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        const expandido = document.querySelector('.expandido');
        if (expandido) {
            expandido.classList.remove('expandido');
            document.body.classList.remove('em-tela-cheia');
            for (let i = 1; i <= 4; i++) {
                const btn = document.getElementById(`btn-expandir-${i}`);
                if (btn) btn.innerText = '⛶';
            }
        }
    }
});

// Inicialização do sistema
window.onload = async () => {
    await listarEComecarCameras();
    await inicializarIA();
};