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
        
        // Começa a analisar os frames da Câmera 1 (Vista Superior)
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

        // Subimos a confiança para 90% para ficar ainda mais rigoroso
        if (nomeClasse === "Bactéria" && probabilidade > 0.90) {
            veBactériaAgora = true;
        }
    }

    // Lógica do Filtro de Estabilidade (Debounce)
    if (veBactériaAgora) {
        contadoresBacteria[numeroBloco]++;
        contadoresVazio[numeroBloco] = 0; // zera o contador de fundo vazio

        // Se a IA viu a bactéria por 5 frames seguidos, confirma a detecção
        if (contadoresBacteria[numeroBloco] >= 5) {
            estadoAtualDetectado[numeroBloco] = true;
        }
    } else {
        contadoresVazio[numeroBloco]++;
        contadoresBacteria[numeroBloco] = 0; // zera o contador de bactéria

        // Se a IA viu o fundo limpo por 5 frames seguidos, confirma que saiu
        if (contadoresVazio[numeroBloco] >= 5) {
            estadoAtualDetectado[numeroBloco] = false;
        }
    }

    // Configuração dos nomes das telas
    const nomesCameras = {
        1: " Vista Superior",
        2: " Lateral Esquerda",
        3: " Lateral Direita",
        4: " Macro"
    };
    const nomeAtual = nomesCameras[numeroBloco] || `Câmera ${numeroBloco}`;

    // Atualiza a tela apenas se o estado estiver travado/estabilizado
    if (estadoAtualDetectado[numeroBloco]) {
        atualizarStatus(numeroBloco, 'online', `${nomeAtual} - Bactéria!`);
        document.getElementById(`captura${numeroBloco}`).innerText = `Última captura: ${obterHoraAtual()}`;
    } else {
        atualizarStatus(numeroBloco, 'online', 'Online');
    }
}

function expandir(bloco) {
    // Só expande se já não estiver expandido
    if (!bloco.classList.contains("expandido")) {
        bloco.classList.add("expandido");
    }
}

function voltar(event, botao) {
    event.stopPropagation(); // MUITO IMPORTANTE: impede que o clique de fechar clique em "expandir" de novo
    botao.parentElement.classList.remove("expandido");
}

document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        const expandido = document.querySelector('.expandido');
        if (expandido) expandido.classList.remove('expandido');
    }
});


window.onload = async () => {
    await listarEComecarCameras();
    setTimeout(inicializarIA, 3000); 
};