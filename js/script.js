let sessaoYuNet = null;
let sessaoSFace = null;

async function carregarModelosFaciais() {
    try {
        console.log("Carregando modelos faciais...");

        sessaoYuNet = await ort.InferenceSession.create(
            "/modelos_faciais/face_detection_yunet_2026may.onnx"
        );

        sessaoSFace = await ort.InferenceSession.create(
            "/modelos_faciais/face_recognition_sface_2021dec_int8.onnx"
        );

        console.log("MODELOS FACIAIS PRONTOS");
        console.log("YuNet:", sessaoYuNet.inputNames, sessaoYuNet.outputNames);
        console.log("SFace:", sessaoSFace.inputNames, sessaoSFace.outputNames);

    } catch (erro) {
        console.error("ERRO AO CARREGAR MODELOS FACIAIS:", erro);
    }
}

let streams = {};

// Associação fixa: cada bloco sempre usa a câmera com esse nome
const MAPEAMENTO_CAMERAS = {
    1: "Logi C270 HD WebCam",  // Vista Superior
    2: "UVC Camera",            // Lateral Esquerda
    3: "GENERAL - UVC",         // Lateral Direita
    4: "ST301"                   // Macro (microscópio)
};

function atualizarStatus(numero, tipo, texto) {
    const elStatus = document.getElementById(`status${numero}`);
    if (elStatus) {
        elStatus.className = `status ${tipo}`;
        elStatus.innerText = texto;
    }
}

function obterHoraAtual() {
    const agora = new Date();
    return agora.toTimeString().split(' ')[0];
}

async function listarEComecarCameras() {
    try {
        // Pede permissão inicial
        const streamInicial = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        streamInicial.getTracks().forEach(track => track.stop());

        // Mapeia todas as câmeras conectadas
        const dispositivos = await navigator.mediaDevices.enumerateDevices();
        const camerasEncontradas = dispositivos.filter(device => device.kind === "videoinput");

        console.log(`Total de câmeras encontradas: ${camerasEncontradas.length}`);

        // Para cada bloco, procura a câmera pelo nome esperado
        for (let numeroBloco = 1; numeroBloco <= 4; numeroBloco++) {
            const nomeEsperado = MAPEAMENTO_CAMERAS[numeroBloco];
            const camera = camerasEncontradas.find(cam => cam.label.includes(nomeEsperado));

            if (!camera) {
                atualizarStatus(numeroBloco, 'offline', 'Sem Câmera');
                document.getElementById(`captura${numeroBloco}`).innerText = "Última captura: --:--:--";
                continue;
            }

            await ligarCamera(numeroBloco, camera.deviceId);
            // Pausa de 1.2 segundos entre a abertura de cada câmera
            await new Promise(resolve => setTimeout(resolve, 1200));
        }

    } catch (erro) {
        console.error("Erro ao listar dispositivos:", erro);
        atualizarStatus(1, 'offline', 'Erro de Acesso');
    }
}

async function ligarCamera(numeroBloco, deviceId) {
    const video = document.getElementById(`camera${numeroBloco}`);
    if (!video) return;

    try {
        atualizarStatus(numeroBloco, 'inicializando', 'Inicializando...');

        // Força resolução baixa (320x240) para não estourar a banda da USB com 3+ câmeras
        const constraints = {
            video: {
                deviceId: { exact: deviceId },
                width: { ideal: 320 },
                height: { ideal: 240 },
                frameRate: { max: 15 }
            },
            audio: false
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);

        streams[numeroBloco] = stream;
        video.srcObject = stream;
        await video.play();

        const nomes = { 1: "Vista Superior", 2: "Lateral Esquerda", 3: "Lateral Direita", 4: "Macro" };
        atualizarStatus(numeroBloco, 'online', `${nomes[numeroBloco]} - Online`);
        document.getElementById(`captura${numeroBloco}`).innerText = `Última captura: ${obterHoraAtual()}`;

    } catch (erro) {
        console.error(`Erro na câmera do Bloco ${numeroBloco}:`, erro);
        atualizarStatus(numeroBloco, 'offline', 'Falha USB/Ocupada');
    }
}

// Integrando o Python YOLOv8
let iaEmExecucao = false;

function inicializarIA() {
    setInterval(loopAnalisePython, 500);
}

async function loopAnalisePython() {
    if (iaEmExecucao) return;
    iaEmExecucao = true;

    for (let i = 1; i <= 4; i++) {
        const video = document.getElementById(`camera${i}`);
        if (video && video.readyState === 4 && streams[i]) {
            await enviarQuadroParaIA(video, i);
        }
    }

    iaEmExecucao = false;
}

async function enviarQuadroParaIA(videoElement, numeroBloco) {
    const canvas = document.createElement('canvas');
    canvas.width = 320; 
    canvas.height = 240;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
        if (!blob) return;

        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        try {
            const resposta = await fetch('http://127.0.0.1:8000/analisar', {
                method: 'POST',
                body: formData
            });

            const dados = await resposta.json();
            const nomesCameras = { 1: "Vista Superior", 2: "Lateral Esquerda", 3: "Lateral Direita", 4: "Macro" };
            const nomeAtual = nomesCameras[numeroBloco] || `Câmera ${numeroBloco}`;

            if (dados.sucesso && dados.deteccoes && dados.deteccoes.length > 0) {
                const detec = dados.deteccoes[0];
                atualizarStatus(
                    numeroBloco, 
                    'online status-bacteria', 
                    `${nomeAtual} - ${detec.classe.toUpperCase()} (${detec.confianca}%)`
                );
            } else {
                atualizarStatus(numeroBloco, 'online', `${nomeAtual} - Online`);
            }
        } catch (err) {
            // Ignora falhas pontuais de conexão com o servidor
        }
    }, 'image/jpeg', 0.7);
}

function capturarImagemManualmente(numeroBloco) {
    const video = document.getElementById(`camera${numeroBloco}`);
    const imgFoto = document.getElementById(`foto${numeroBloco}`);
    
    if (!video || !streams[numeroBloco]) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    imgFoto.src = canvas.toDataURL('image/png');
    imgFoto.style.display = 'block';
    document.getElementById(`captura${numeroBloco}`).innerText = `Última captura: ${obterHoraAtual()}`;
}

function alternarExpandirCamera(numeroBloco) {
    const bloco = document.getElementById(`bloco${numeroBloco}`);
    const btn = document.getElementById(`btn-expandir-${numeroBloco}`);
    
    if (!bloco || !btn) return;

    const jaEstaExpandido = bloco.classList.contains('expandido');

    if (jaEstaExpandido) {
        bloco.classList.remove('expandido');
        document.body.classList.remove('em-tela-cheia');
        btn.innerText = '⛶';
    } else {
        document.querySelectorAll('.bloco').forEach(b => b.classList.remove('expandido'));
        for (let i = 1; i <= 4; i++) {
            const b = document.getElementById(`btn-expandir-${i}`);
            if (b) b.innerText = '⛶';
        }
        bloco.classList.add('expandido');
        document.body.classList.add('em-tela-cheia');
        btn.innerText = '🗗';
    }
}

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

window.onload = async () => {
    await carregarModelosFaciais();
    await listarEComecarCameras();
    inicializarIA();
};