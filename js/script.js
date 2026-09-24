let sessaoYuNet = null;
let sessaoSFace = null;

// Dicionário global para armazenar estado mais recente das análises de IA por câmera
let estadoAnaliseIAPorCamera = {
    1: { deteccao: "Sem alterações", confianca: 100 },
    2: { deteccao: "Sem alterações", confianca: 100 },
    3: { deteccao: "Sem alterações", confianca: 100 },
    4: { deteccao: "Sem alterações", confianca: 100 }
};

let bancoHistorico = JSON.parse(localStorage.getItem('kairos_banco_historico')) || [];

// Limite de retenção em milissegundos (7 minutos)
const TEMPO_EXPIRACAO_MS = 7 * 60 * 1000;

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
    } catch (erro) {
        console.error("ERRO AO CARREGAR MODELOS FACIAIS:", erro);
    }
}

let streams = {};

const MAPEAMENTO_CAMERAS = {
    1: "Logi C270 HD WebCam",
    2: "UVC Camera",
    3: "GENERAL - UVC",
    4: "ST301"
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
        const streamInicial = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        streamInicial.getTracks().forEach(track => track.stop());

        const dispositivos = await navigator.mediaDevices.enumerateDevices();
        const camerasEncontradas = dispositivos.filter(device => device.kind === "videoinput");

        for (let numeroBloco = 1; numeroBloco <= 4; numeroBloco++) {
            const nomeEsperado = MAPEAMENTO_CAMERAS[numeroBloco];
            const camera = camerasEncontradas.find(cam => cam.label.includes(nomeEsperado));

            if (!camera) {
                atualizarStatus(numeroBloco, 'offline', 'Sem Câmera');
                document.getElementById(`captura${numeroBloco}`).innerText = "Última captura: --:--:--";
                continue;
            }

            await ligarCamera(numeroBloco, camera.deviceId);
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

// ============================================================
// INTEGRAÇÃO COM O SERVIDOR PYTHON
// ============================================================
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
            await reconhecerPessoa(video, i);
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

                estadoAnaliseIAPorCamera[numeroBloco] = {
                    deteccao: detec.classe.toUpperCase(),
                    confianca: detec.confianca
                };
            } else {
                atualizarStatus(numeroBloco, 'online', `${nomeAtual} - Online`);
                estadoAnaliseIAPorCamera[numeroBloco] = {
                    deteccao: "Sem alterações",
                    confianca: 100
                };
            }
        } catch (err) {
            // Ignora falhas pontuais
        }
    }, 'image/jpeg', 0.7);
}

// ============================================================
// RECONHECIMENTO FACIAL KAIRÓS
// ============================================================

function criarCanvasImagem(imageData) {
    const canvas = document.createElement("canvas");
    canvas.width = imageData.width;
    canvas.height = imageData.height;

    const contexto = canvas.getContext("2d");
    contexto.putImageData(imageData, 0, 0);

    return canvas;
}

function selecionarMelhorRosto(deteccoes) {
    if (!deteccoes || deteccoes.length === 0) return null;

    let melhorRosto = deteccoes[0];
    for (const deteccao of deteccoes) {
        if (deteccao[4] > melhorRosto[4]) {
            melhorRosto = deteccao;
        }
    }
    return melhorRosto;
}

async function detectarRostoYuNet(imageData) {
    const largura = imageData.width;
    const altura = imageData.height;
    const dados = imageData.data;
    const tamanho = largura * altura;

    const tensorData = new Float32Array(3 * tamanho);

    let indiceB = 0;
    let indiceG = tamanho;
    let indiceR = tamanho * 2;

    for (let i = 0; i < dados.length; i += 4) {
        tensorData[indiceB++] = dados[i + 2];
        tensorData[indiceG++] = dados[i + 1];
        tensorData[indiceR++] = dados[i];
    }

    const tensor = new ort.Tensor("float32", tensorData, [1, 3, altura, largura]);
    return await sessaoYuNet.run({ input: tensor });
}

function decodificarDeteccoesYuNet(resultado, larguraImagem, alturaImagem) {
    const deteccoes = [];
    const strides = [8, 16, 32];

    for (const stride of strides) {
        const cls = resultado[`cls_${stride}`].data;
        const obj = resultado[`obj_${stride}`].data;
        const bbox = resultado[`bbox_${stride}`].data;
        const kps = resultado[`kps_${stride}`].data;

        const quantidade = cls.length;
        const larguraMapa = Math.ceil(larguraImagem / stride);

        for (let i = 0; i < quantidade; i++) {
            const scoreClasse = cls[i];
            const scoreObjeto = obj[i];
            const score = Math.sqrt(scoreClasse * scoreObjeto);

            if (score < 0.5) continue;

            const coluna = i % larguraMapa;
            const linha = Math.floor(i / larguraMapa);

            const centroX = coluna * stride;
            const centroY = linha * stride;

            const indiceBBox = i * 4;
            const indiceKps = i * 10;

            const centroRostoX = bbox[indiceBBox] * stride + centroX;
            const centroRostoY = bbox[indiceBBox + 1] * stride + centroY;

            const larguraRosto = Math.exp(bbox[indiceBBox + 2]) * stride;
            const alturaRosto = Math.exp(bbox[indiceBBox + 3]) * stride;

            const x = centroRostoX - larguraRosto / 2;
            const y = centroRostoY - alturaRosto / 2;

            deteccoes.push([
                x, y, larguraRosto, alturaRosto, score,
                centroX + kps[indiceKps] * stride,
                centroY + kps[indiceKps + 1] * stride,
                centroX + kps[indiceKps + 2] * stride,
                centroY + kps[indiceKps + 3] * stride,
                centroX + kps[indiceKps + 4] * stride,
                centroY + kps[indiceKps + 5] * stride,
                centroX + kps[indiceKps + 6] * stride,
                centroY + kps[indiceKps + 7] * stride,
                centroX + kps[indiceKps + 8] * stride,
                centroY + kps[indiceKps + 9] * stride
            ]);
        }
    }
    return deteccoes;
}

function recortarRosto(imageData, deteccao) {
    if (!deteccao || deteccao.length < 15) return null;

    const canvasOrigem = criarCanvasImagem(imageData);

    const x = Math.max(0, Math.floor(deteccao[0]));
    const y = Math.max(0, Math.floor(deteccao[1]));

    const largura = Math.floor(deteccao[2]);
    const altura = Math.floor(deteccao[3]);

    const olhoDireito = { x: deteccao[5], y: deteccao[6] };
    const olhoEsquerdo = { x: deteccao[7], y: deteccao[8] };

    const margemX = largura * 0.25;
    const margemY = altura * 0.30;

    const origemX = Math.max(0, x - margemX);
    const origemY = Math.max(0, y - margemY);

    const origemLargura = Math.min(imageData.width - origemX, largura + margemX * 2);
    const origemAltura = Math.min(imageData.height - origemY, altura + margemY * 2);

    const canvasRecorte = document.createElement("canvas");
    canvasRecorte.width = 112;
    canvasRecorte.height = 112;

    const contexto = canvasRecorte.getContext("2d");

    contexto.drawImage(
        canvasOrigem,
        origemX, origemY, origemLargura, origemAltura,
        0, 0, 112, 112
    );

    const escalaX = 112 / origemLargura;
    const escalaY = 112 / origemAltura;

    const olhoDireitoLocal = {
        x: (olhoDireito.x - origemX) * escalaX,
        y: (olhoDireito.y - origemY) * escalaY
    };

    const olhoEsquerdoLocal = {
        x: (olhoEsquerdo.x - origemX) * escalaX,
        y: (olhoEsquerdo.y - origemY) * escalaY
    };

    const dx = olhoEsquerdoLocal.x - olhoDireitoLocal.x;
    const dy = olhoEsquerdoLocal.y - olhoDireitoLocal.y;

    const angulo = Math.atan2(dy, dx);

    const centroX = 56;
    const centroY = 56;

    contexto.clearRect(0, 0, 112, 112);
    contexto.save();
    contexto.translate(centroX, centroY);
    contexto.rotate(-angulo);
    contexto.translate(-centroX, -centroY);

    contexto.drawImage(
        canvasOrigem,
        origemX, origemY, origemLargura, origemAltura,
        0, 0, 112, 112
    );

    contexto.restore();

    return contexto.getImageData(0, 0, 112, 112);
}

async function gerarEmbeddingSFace(imageData) {
    const dados = imageData.data;
    const tamanho = 112 * 112;

    const tensorData = new Float32Array(3 * tamanho);

    let indiceR = 0;
    let indiceG = tamanho;
    let indiceB = tamanho * 2;

    for (let i = 0; i < dados.length; i += 4) {
        tensorData[indiceR++] = dados[i];
        tensorData[indiceG++] = dados[i + 1];
        tensorData[indiceB++] = dados[i + 2];
    }

    const tensor = new ort.Tensor("float32", tensorData, [1, 3, 112, 112]);
    const resultado = await sessaoSFace.run({ data: tensor });

    return Array.from(resultado.fc1.data);
}

async function reconhecerPessoa(videoElement, numeroBloco) {
    try {
        if (!sessaoYuNet || !sessaoSFace || !videoElement.videoWidth || !videoElement.videoHeight) return;

        const canvas = document.createElement("canvas");
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;

        const contexto = canvas.getContext("2d");
        contexto.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

        const imageData = contexto.getImageData(0, 0, canvas.width, canvas.height);
        const resultadoYuNet = await detectarRostoYuNet(imageData);
        const deteccoes = decodificarDeteccoesYuNet(resultadoYuNet, imageData.width, imageData.height);
        const melhorRosto = selecionarMelhorRosto(deteccoes);

        if (!melhorRosto) return;

        const rostoAlinhado = recortarRosto(imageData, melhorRosto);
        if (!rostoAlinhado) return;

        const embedding = await gerarEmbeddingSFace(rostoAlinhado);
        if (embedding.length !== 128) return;

        const resposta = await fetch("http://127.0.0.1:8000/reconhecer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ embedding: embedding })
        });

        const resultado = await resposta.json();

        if (resultado.sucesso && resultado.reconhecido && resultado.usuario) {
            console.log(`✅ CÂMERA ${numeroBloco}:`, resultado.usuario.nome, resultado.similaridade);
        }
    } catch (erro) {
        console.error(`ERRO NO RECONHECIMENTO DA CÂMERA ${numeroBloco}:`, erro);
    }
}

// ============================================================
// HISTÓRICO / PACIENTE / OPERADOR / EXCLUSÃO AUTOMÁTICA (7 MIN)
// ============================================================

function obterOperadorAtual() {
    const inputOp = document.getElementById('nome-operador');
    if (inputOp && inputOp.value.trim() !== '') {
        return inputOp.value.trim();
    }
    return sessionStorage.getItem('operador') || "Farmacêutico Responsável";
}

function salvarOperador() {
    const inputOp = document.getElementById('nome-operador');
    if (inputOp) {
        sessionStorage.setItem('operador', inputOp.value);
    }
}

function obterPacienteAtual() {
    const inputPac = document.getElementById('nome-paciente');
    if (inputPac && inputPac.value.trim() !== '') {
        return inputPac.value.trim();
    }
    return sessionStorage.getItem('paciente') || "Não Identificado";
}

function salvarPaciente() {
    const inputPac = document.getElementById('nome-paciente');
    if (inputPac) {
        sessionStorage.setItem('paciente', inputPac.value);
    }
}

function abrirModalHistorico() {
    const modal = document.getElementById('modal-historico');
    if (modal) {
        modal.classList.remove('modal-oculta');
        limparRegistrosAntigos();
        renderizarTabelaHistorico();
    }
}

function fecharModalHistorico() {
    const modal = document.getElementById('modal-historico');
    if (modal) {
        modal.classList.add('modal-oculta');
    }
}

// EXCLUSÃO AUTOMÁTICA DE REGISTROS/FOTOS COM MAIS DE 7 MINUTOS
function limparRegistrosAntigos() {
    const agora = Date.now();
    const totalAntes = bancoHistorico.length;

    bancoHistorico = bancoHistorico.filter(item => {
        return (agora - item.timestamp) < TEMPO_EXPIRACAO_MS;
    });

    if (bancoHistorico.length !== totalAntes) {
        localStorage.setItem('kairos_banco_historico', JSON.stringify(bancoHistorico));
        
        const modal = document.getElementById('modal-historico');
        if (modal && !modal.classList.contains('modal-oculta')) {
            renderizarTabelaHistorico();
        }
    }
}

function iniciarCapturaAutomaticaEmLote() {
    setInterval(() => {
        capturarLoteGeral();
    }, 15000);

    // Verificação de limpeza automática a cada 5 segundos
    setInterval(() => {
        limparRegistrosAntigos();
    }, 5000);
}

function capturarLoteGeral() {
    limparRegistrosAntigos();

    const agora = Date.now();
    const hora = obterHoraAtual();
    const operador = obterOperadorAtual();
    const paciente = obterPacienteAtual();
    const nomesCameras = { 
        1: "Vista Superior", 
        2: "Lateral Esquerda", 
        3: "Lateral Direita", 
        4: "Macro" 
    };

    for (let i = 1; i <= 4; i++) {
        const video = document.getElementById(`camera${i}`);
        let fotoDataUrl = null;
        let statusCaptura = "Sucesso";

        if (video && streams[i] && video.readyState === 4) {
            try {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth || 320;
                canvas.height = video.videoHeight || 240;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                fotoDataUrl = canvas.toDataURL('image/jpeg', 0.6);
            } catch (e) {
                statusCaptura = "Falha na Captura";
            }
        } else {
            statusCaptura = "Falha na Captura";
        }

        const infoIA = estadoAnaliseIAPorCamera[i] || { deteccao: "Sem alterações", confianca: 100 };

        const registro = {
            id: agora + i,
            timestamp: agora,
            horario: hora,
            paciente: paciente,
            camera: nomesCameras[i],
            foto: fotoDataUrl,
            status: statusCaptura,
            deteccao: statusCaptura === "Falha na Captura" ? "Câmera Indisponível" : infoIA.deteccao,
            confianca: statusCaptura === "Falha na Captura" ? 0 : infoIA.confianca,
            operador: operador
        };

        bancoHistorico.unshift(registro);
    }

    localStorage.setItem('kairos_banco_historico', JSON.stringify(bancoHistorico));

    const modal = document.getElementById('modal-historico');
    if (modal && !modal.classList.contains('modal-oculta')) {
        renderizarTabelaHistorico();
    }
}

function renderizarTabelaHistorico() {
    const corpo = document.getElementById('corpo-tabela-amostras');
    if (!corpo) return;

    corpo.innerHTML = '';

    if (bancoHistorico.length === 0) {
        corpo.innerHTML = `<tr><td colspan="7" style="text-align:center; color:#94a3b8; padding:20px;">Nenhum registro ativo nos últimos 7 minutos.</td></tr>`;
        return;
    }

    bancoHistorico.forEach(item => {
        const tr = document.createElement('tr');

        let colunaFoto = '';
        if (item.status === "Falha na Captura" || !item.foto) {
            colunaFoto = `<span class="badge-confianca badge-falha">Falha na Captura</span>`;
        } else {
            colunaFoto = `<img src="${item.foto}" class="img-tb" alt="Amostra">`;
        }

        let badgeConfianca = '';
        if (item.status === "Falha na Captura") {
            badgeConfianca = `<span class="badge-confianca badge-falha">0%</span>`;
        } else if (item.confianca >= 70) {
            badgeConfianca = `<span class="badge-confianca badge-alta">${item.confianca}%</span>`;
        } else {
            badgeConfianca = `<span class="badge-confianca badge-media">${item.confianca}%</span>`;
        }

        tr.innerHTML = `
            <td>${colunaFoto}</td>
            <td><strong>${item.horario}</strong></td>
            <td><strong>${item.paciente || "Não Identificado"}</strong></td>
            <td>${item.camera}</td>
            <td>${item.deteccao}</td>
            <td>${badgeConfianca}</td>
            <td>${item.operador}</td>
        `;

        corpo.appendChild(tr);
    });
}

// ============================================================
// CONTROLES MANUAIS E EXPANSÃO
// ============================================================

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

document.addEventListener('keydown', function (e) {
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

// ============================================================
// INICIALIZAÇÃO VIA TELA DE ENTRADA (LANDING SCREEN)
// ============================================================

async function iniciarSistemaKairos() {
    const btnIniciar = document.getElementById('btn-iniciar-sistema');
    const loader = document.getElementById('landing-status-loader');
    const landingScreen = document.getElementById('landing-screen');

    if (btnIniciar) btnIniciar.style.display = 'none';
    if (loader) loader.classList.remove('modal-oculta');

    try {
        await carregarModelosFaciais();
        await listarEComecarCameras();
        inicializarIA();
        iniciarCapturaAutomaticaEmLote();

        if (landingScreen) {
            landingScreen.classList.add('fade-out');
            setTimeout(() => {
                landingScreen.style.display = 'none';
            }, 800);
        }
    } catch (erro) {
        console.error("Erro durante a inicialização do sistema KAIROS:", erro);
        if (loader) loader.innerText = "Erro ao conectar aos componentes. Verifique as permissões de vídeo.";
    }
}

window.onload = () => {
    const opSalvo = sessionStorage.getItem('operador');
    if (opSalvo) {
        const inputOp = document.getElementById('nome-operador');
        if (inputOp) inputOp.value = opSalvo;
    }

    const pacSalvo = sessionStorage.getItem('paciente');
    if (pacSalvo) {
        const inputPac = document.getElementById('nome-paciente');
        if (inputPac) inputPac.value = pacSalvo;
    }
};