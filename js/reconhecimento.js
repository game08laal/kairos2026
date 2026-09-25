console.log("===== KAIROS | RECONHECIMENTO FACIAL =====");

const video = document.getElementById("camera-rosto");
const mensagem = document.getElementById("mensagem");
const status = document.getElementById("status-reconhecimento");
const operadorReconhecido = document.getElementById("operador-reconhecido");
const nomeOperador = document.getElementById("nome-operador");
const btnContinuar = document.getElementById("btn-continuar");

let sessaoYuNet = null;
let sessaoSFace = null;

let detectando = false;
let processando = false;
let reconhecimentoConcluido = false;

let ultimoReconhecimento = 0;

const INTERVALO_RECONHECIMENTO = 2500;

const URL_FASTAPI = "http://127.0.0.1:8000/reconhecer";

const URL_YUNET =
    "/modelos_faciais/face_detection_yunet_2026may.onnx";

const URL_SFACE =
    "/modelos_faciais/face_recognition_sface_2021dec_int8.onnx";


// ============================================================
// AUXILIARES
// ============================================================

function atualizarStatus(texto, classe = "aguardando") {

    if (!status) {
        return;
    }

    status.innerText = texto;
    status.className = `status ${classe}`;
}


function esperar(ms) {

    return new Promise(resolve => {
        setTimeout(resolve, ms);
    });
}


// ============================================================
// CÂMERA
// ============================================================

async function iniciarCamera() {

    try {

        console.log("1 - Iniciando câmera...");

        const stream =
            await navigator.mediaDevices.getUserMedia({

                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: "user"
                },

                audio: false
            });

        video.srcObject = stream;

        await video.play();

        console.log("2 - Câmera iniciada.");

        atualizarStatus(
            "Câmera pronta. Carregando reconhecimento facial.",
            "aguardando"
        );

    } catch (erro) {

        console.error(
            "ERRO AO ACESSAR CÂMERA:",
            erro
        );

        atualizarStatus(
            "Não foi possível acessar a câmera.",
            "erro"
        );
    }
}


// ============================================================
// CARREGAR MODELOS
// ============================================================

async function carregarModelosFaciais() {

    try {

        console.log("3 - Carregando modelos...");


        // --------------------------------------------------------
        // YUNET
        // --------------------------------------------------------

        console.log("4 - Carregando YuNet...");

        sessaoYuNet =
            await ort.InferenceSession.create(
                URL_YUNET
            );

        console.log("5 - YuNet carregado.");

        console.log(
            "ENTRADAS YUNET:",
            sessaoYuNet.inputNames
        );

        console.log(
            "SAÍDAS YUNET:",
            sessaoYuNet.outputNames
        );


        // --------------------------------------------------------
        // SFACE
        // --------------------------------------------------------

        console.log("6 - Carregando SFace...");

        sessaoSFace =
            await ort.InferenceSession.create(
                URL_SFACE
            );

        console.log("7 - SFace carregado.");

        console.log(
            "ENTRADA SFACE:",
            sessaoSFace.inputNames
        );

        console.log(
            "SAÍDA SFACE:",
            sessaoSFace.outputNames
        );


        atualizarStatus(
            "Reconhecimento facial pronto. Posicione seu rosto.",
            "sucesso"
        );

        console.log(
            "8 - MODELOS FACIAIS PRONTOS."
        );

    } catch (erro) {

        console.error(
            "ERRO AO CARREGAR MODELOS:",
            erro
        );

        atualizarStatus(
            "Erro ao carregar os modelos faciais.",
            "erro"
        );

        throw erro;
    }
}


// ============================================================
// CAPTURAR FRAME
// ============================================================

function capturarFrame() {

    if (
        !video.videoWidth ||
        !video.videoHeight
    ) {
        return null;
    }

    const canvas =
        document.createElement("canvas");

    canvas.width =
        video.videoWidth;

    canvas.height =
        video.videoHeight;

    const contexto =
        canvas.getContext("2d", {
            willReadFrequently: true
        });

    contexto.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );

    return canvas;
}


// ============================================================
// PREPARAR IMAGEM PARA YUNET
// ============================================================

function prepararImagemYuNet(canvas) {

    const largura =
        canvas.width;

    const altura =
        canvas.height;

    const contexto =
        canvas.getContext("2d", {
            willReadFrequently: true
        });

    const imagem =
        contexto.getImageData(
            0,
            0,
            largura,
            altura
        );

    const quantidadePixels =
        largura * altura;

    const dados =
        new Float32Array(
            3 * quantidadePixels
        );

    for (
        let y = 0;
        y < altura;
        y++
    ) {

        for (
            let x = 0;
            x < largura;
            x++
        ) {

            const pixel =
                (y * largura + x) * 4;

            const posicao =
                y * largura + x;


            // B
            dados[posicao] =
                imagem.data[pixel + 2];


            // G
            dados[
                quantidadePixels + posicao
            ] =
                imagem.data[pixel + 1];


            // R
            dados[
                2 * quantidadePixels + posicao
            ] =
                imagem.data[pixel];
        }
    }

    return new ort.Tensor(
        "float32",
        dados,
        [1, 3, altura, largura]
    );
}


// ============================================================
// EXECUTAR YUNET
// ============================================================

async function executarYuNet() {

    const canvas =
        capturarFrame();

    if (!canvas) {
        return null;
    }

    const tensor =
        prepararImagemYuNet(canvas);

    const nomeEntrada =
        sessaoYuNet.inputNames[0];

    const resultado =
        await sessaoYuNet.run({

            [nomeEntrada]:
                tensor

        });

    return {

        resultado,

        larguraImagem:
            canvas.width,

        alturaImagem:
            canvas.height,

        canvas
    };
}


// ============================================================
// INTERPRETAR YUNET
// ============================================================

function interpretarYuNet(resultado) {

    const saidas =
        resultado.resultado;

    const largura =
        resultado.larguraImagem;

    const altura =
        resultado.alturaImagem;

    const deteccoes = [];

    const strides = [
        8,
        16,
        32
    ];


    for (const stride of strides) {

        const nomeCls =
            `cls_${stride}`;

        const nomeObj =
            `obj_${stride}`;

        const nomeBbox =
            `bbox_${stride}`;

        const nomeKps =
            `kps_${stride}`;


        if (
            !saidas[nomeCls] ||
            !saidas[nomeObj] ||
            !saidas[nomeBbox] ||
            !saidas[nomeKps]
        ) {

            console.error(
                "Saída YuNet ausente:",
                stride
            );

            continue;
        }


        const cls =
            saidas[nomeCls].data;

        const obj =
            saidas[nomeObj].data;

        const bbox =
            saidas[nomeBbox].data;

        const kps =
            saidas[nomeKps].data;


        const larguraMapa =
            Math.ceil(largura / stride);

        const alturaMapa =
            Math.ceil(altura / stride);

        const quantidadeEsperada =
            larguraMapa *
            alturaMapa;

        const quantidade =
            Math.min(
                cls.length,
                quantidadeEsperada
            );


        for (
            let i = 0;
            i < quantidade;
            i++
        ) {

            let scoreClasse =
                Number(cls[i]);

            let scoreObjeto =
                Number(obj[i]);


            scoreClasse =
                Math.max(
                    0,
                    Math.min(
                        1,
                        scoreClasse
                    )
                );


            scoreObjeto =
                Math.max(
                    0,
                    Math.min(
                        1,
                        scoreObjeto
                    )
                );


            const confianca =
                Math.sqrt(
                    scoreClasse *
                    scoreObjeto
                );


            if (
                confianca < 0.50
            ) {
                continue;
            }


            const coluna =
                i % larguraMapa;

            const linha =
                Math.floor(
                    i / larguraMapa
                );


            const centroX =
                (
                    coluna +
                    Number(bbox[i * 4])
                ) *
                stride;


            const centroY =
                (
                    linha +
                    Number(bbox[i * 4 + 1])
                ) *
                stride;


            const larguraRosto =
                Math.exp(
                    Number(
                        bbox[i * 4 + 2]
                    )
                ) *
                stride;


            const alturaRosto =
                Math.exp(
                    Number(
                        bbox[i * 4 + 3]
                    )
                ) *
                stride;


            const x =
                centroX -
                larguraRosto / 2;


            const y =
                centroY -
                alturaRosto / 2;


            // ----------------------------------------------------
            // 5 LANDMARKS DO YUNET
            //
            // 0 = olho direito
            // 1 = olho esquerdo
            // 2 = nariz
            // 3 = canto direito da boca
            // 4 = canto esquerdo da boca
            // ----------------------------------------------------

            const pontos = [];


            for (
                let ponto = 0;
                ponto < 5;
                ponto++
            ) {

                const indice =
                    i * 10 +
                    ponto * 2;


                const pontoX =
                    (
                        Number(
                            kps[indice]
                        ) +
                        coluna
                    ) *
                    stride;


                const pontoY =
                    (
                        Number(
                            kps[indice + 1]
                        ) +
                        linha
                    ) *
                    stride;


                pontos.push({

                    x: pontoX,
                    y: pontoY

                });
            }


            if (
                !Number.isFinite(x) ||
                !Number.isFinite(y) ||
                !Number.isFinite(larguraRosto) ||
                !Number.isFinite(alturaRosto)
            ) {
                continue;
            }


            deteccoes.push({

                x,
                y,

                largura:
                    larguraRosto,

                altura:
                    alturaRosto,

                confianca,

                pontos
            });
        }
    }


    return deteccoes;
}
// ============================================================
// REMOVER DETECÇÕES DUPLICADAS
// ============================================================

function calcularIoU(a, b) {

    const esquerda =
        Math.max(
            a.x,
            b.x
        );

    const topo =
        Math.max(
            a.y,
            b.y
        );

    const direita =
        Math.min(
            a.x + a.largura,
            b.x + b.largura
        );

    const baixo =
        Math.min(
            a.y + a.altura,
            b.y + b.altura
        );


    const largura =
        Math.max(
            0,
            direita - esquerda
        );

    const altura =
        Math.max(
            0,
            baixo - topo
        );

    const intersecao =
        largura * altura;


    const areaA =
        a.largura *
        a.altura;

    const areaB =
        b.largura *
        b.altura;


    const uniao =
        areaA +
        areaB -
        intersecao;


    if (
        uniao <= 0
    ) {
        return 0;
    }


    return (
        intersecao /
        uniao
    );
}


function aplicarNMS(deteccoes) {

    const ordenadas =
        [...deteccoes].sort(
            (a, b) =>
                b.confianca -
                a.confianca
        );


    const mantidas = [];


    for (
        const deteccao
        of ordenadas
    ) {

        let sobreposta =
            false;


        for (
            const mantida
            of mantidas
        ) {

            if (
                calcularIoU(
                    deteccao,
                    mantida
                ) > 0.30
            ) {

                sobreposta =
                    true;

                break;
            }
        }


        if (!sobreposta) {

            mantidas.push(
                deteccao
            );
        }
    }


    return mantidas;
}


// ============================================================
// ESCOLHER MELHOR ROSTO
// ============================================================

function selecionarMelhorRosto(
    deteccoes
) {

    if (
        !deteccoes ||
        !deteccoes.length
    ) {
        return null;
    }


    const semDuplicatas =
        aplicarNMS(
            deteccoes
        );


    let melhor =
        semDuplicatas[0];


    for (
        const deteccao
        of semDuplicatas
    ) {

        if (
            deteccao.confianca >
            melhor.confianca
        ) {

            melhor =
                deteccao;
        }
    }


    return melhor;
}


// ============================================================
// CONVERTER DETECÇÃO PARA O FORMATO DA SOFIA
// ============================================================

function converterDeteccaoParaFormatoSofia(deteccao) {

    if (!deteccao) {
        console.error(
            "Detecção facial inexistente."
        );

        return null;
    }


    console.log(
        "OBJETO COMPLETO DA DETECÇÃO:",
        deteccao
    );

    console.log(
        "CHAVES DA DETECÇÃO:",
        Object.keys(deteccao)
    );

    console.log(
        "VALORES DA DETECÇÃO:",
        Object.values(deteccao)
    );


    const x =
        deteccao.x !== undefined
            ? deteccao.x
            : (
                deteccao.bbox
                    ? deteccao.bbox[0]
                    : 0
            );


    const y =
        deteccao.y !== undefined
            ? deteccao.y
            : (
                deteccao.bbox
                    ? deteccao.bbox[1]
                    : 0
            );


    const largura =
        deteccao.largura !== undefined
            ? deteccao.largura
            : (
                deteccao.bbox
                    ? deteccao.bbox[2]
                    : 0
            );


    const altura =
        deteccao.altura !== undefined
            ? deteccao.altura
            : (
                deteccao.bbox
                    ? deteccao.bbox[3]
                    : 0
            );


    const confianca =
        deteccao.confianca !== undefined
            ? deteccao.confianca
            : (deteccao.score || 1.0);


    /*
     * Mapeia dinamicamente onde estão os 5 pontos faciais
     */

    let pontos =
        deteccao.pontos ||
        deteccao.pontosFaciais ||
        deteccao.landmarks ||
        deteccao.keypoints ||
        deteccao.kps;


    if (!pontos) {

        console.error(
            "Não encontrei os pontos faciais dentro da detecção."
        );

        console.error(
            "Propriedades disponíveis:",
            Object.keys(deteccao)
        );

        return null;
    }


    /*
     * Esperamos 5 pontos:
     * 0 = olho direito
     * 1 = olho esquerdo
     * 2 = nariz
     * 3 = boca direita
     * 4 = boca esquerda
     */

    if (pontos.length < 5) {

        console.error(
            "Quantidade insuficiente de pontos faciais:",
            pontos
        );

        return null;
    }


    const olhoDireito =
        pontos[0];

    const olhoEsquerdo =
        pontos[1];

    const nariz =
        pontos[2];

    const bocaDireita =
        pontos[3];

    const bocaEsquerda =
        pontos[4];


    return [

        // Caixa do rosto
        x,
        y,
        largura,
        altura,

        // Confiança
        confianca,

        // Olho direito
        olhoDireito.x !== undefined
            ? olhoDireito.x
            : olhoDireito[0],

        olhoDireito.y !== undefined
            ? olhoDireito.y
            : olhoDireito[1],

        // Olho esquerdo
        olhoEsquerdo.x !== undefined
            ? olhoEsquerdo.x
            : olhoEsquerdo[0],

        olhoEsquerdo.y !== undefined
            ? olhoEsquerdo.y
            : olhoEsquerdo[1],

        // Nariz
        nariz.x !== undefined
            ? nariz.x
            : nariz[0],

        nariz.y !== undefined
            ? nariz.y
            : nariz[1],

        // Boca direita
        bocaDireita.x !== undefined
            ? bocaDireita.x
            : bocaDireita[0],

        bocaDireita.y !== undefined
            ? bocaDireita.y
            : bocaDireita[1],

        // Boca esquerda
        bocaEsquerda.x !== undefined
            ? bocaEsquerda.x
            : bocaEsquerda[0],

        bocaEsquerda.y !== undefined
            ? bocaEsquerda.y
            : bocaEsquerda[1]
    ];
}
// ============================================================
// CRIAR CANVAS A PARTIR DE IMAGEDATA
// ============================================================

function criarCanvasImagem(imageData) {

    const canvas =
        document.createElement("canvas");

    canvas.width =
        imageData.width;

    canvas.height =
        imageData.height;

    const contexto =
        canvas.getContext("2d");

    contexto.putImageData(
        imageData,
        0,
        0
    );

    return canvas;
}


// ============================================================
// RECORTE FACIAL
// ============================================================

function recortarRosto(
    imageData,
    deteccao
) {

    if (
        !deteccao ||
        deteccao.length < 15
    ) {

        console.log(
            "Nenhum rosto encontrado."
        );

        return null;
    }


    const canvasOrigem =
        criarCanvasImagem(
            imageData
        );


    const x =
        Math.max(
            0,
            Math.floor(
                deteccao[0]
            )
        );


    const y =
        Math.max(
            0,
            Math.floor(
                deteccao[1]
            )
        );


    const largura =
        Math.floor(
            deteccao[2]
        );


    const altura =
        Math.floor(
            deteccao[3]
        );


    const olhoDireito = {

        x:
            deteccao[5],

        y:
            deteccao[6]

    };


    const olhoEsquerdo = {

        x:
            deteccao[7],

        y:
            deteccao[8]

    };


    const nariz = {

        x:
            deteccao[9],

        y:
            deteccao[10]

    };


    const margemX =
        largura * 0.25;


    const margemY =
        altura * 0.30;


    const origemX =
        Math.max(
            0,
            x - margemX
        );


    const origemY =
        Math.max(
            0,
            y - margemY
        );


    const origemLargura =
        Math.min(
            imageData.width - origemX,
            largura + margemX * 2
        );


    const origemAltura =
        Math.min(
            imageData.height - origemY,
            altura + margemY * 2
        );


    const canvasRecorte =
        document.createElement(
            "canvas"
        );


    canvasRecorte.width =
        112;

    canvasRecorte.height =
        112;


    const contexto =
        canvasRecorte.getContext(
            "2d"
        );


    contexto.drawImage(

        canvasOrigem,

        origemX,
        origemY,
        origemLargura,
        origemAltura,

        0,
        0,
        112,
        112
    );


    const escalaX =
        112 /
        origemLargura;


    const escalaY =
        112 /
        origemAltura;


    const olhoDireitoLocal = {

        x:
            (
                olhoDireito.x -
                origemX
            ) *
            escalaX,

        y:
            (
                olhoDireito.y -
                origemY
            ) *
            escalaY
    };


    const olhoEsquerdoLocal = {

        x:
            (
                olhoEsquerdo.x -
                origemX
            ) *
            escalaX,

        y:
            (
                olhoEsquerdo.y -
                origemY
            ) *
            escalaY
    };


    const narizLocal = {

        x:
            (
                nariz.x -
                origemX
            ) *
            escalaX,

        y:
            (
                nariz.y -
                origemY
            ) *
            escalaY
    };


    const dx =
        olhoEsquerdoLocal.x -
        olhoDireitoLocal.x;


    const dy =
        olhoEsquerdoLocal.y -
        olhoDireitoLocal.y;


    const angulo =
        Math.atan2(
            dy,
            dx
        );


    const centroX =
        56;


    const centroY =
        56;


    contexto.clearRect(
        0,
        0,
        112,
        112
    );


    contexto.save();


    contexto.translate(
        centroX,
        centroY
    );


    contexto.rotate(
        -angulo
    );


    contexto.translate(
        -centroX,
        -centroY
    );


    contexto.drawImage(

        canvasOrigem,

        origemX,
        origemY,
        origemLargura,
        origemAltura,

        0,
        0,
        112,
        112
    );


    contexto.restore();


    const resultado =
        contexto.getImageData(
            0,
            0,
            112,
            112
        );


    return resultado;
}


// ============================================================
// PREPARAR IMAGEM PARA SFACE
// ============================================================

function prepararImagemSFace(
    imageData
) {

    const dados =
        imageData.data;


    const tamanho =
        112 * 112;


    const tensorData =
        new Float32Array(
            3 * tamanho
        );


    let indiceR =
        0;

    let indiceG =
        tamanho;

    let indiceB =
        tamanho * 2;


    for (
        let i = 0;
        i < dados.length;
        i += 4
    ) {

        tensorData[indiceR++] =
            dados[i];

        tensorData[indiceG++] =
            dados[i + 1];

        tensorData[indiceB++] =
            dados[i + 2];
    }


    return new ort.Tensor(

        "float32",

        tensorData,

        [
            1,
            3,
            112,
            112
        ]
    );
}


// ============================================================
// GERAR BIOMETRIA SFACE
// ============================================================

async function gerarBiometriaSFace(
    imageData
) {

    console.log(
        "10 - Preparando rosto para SFace..."
    );


    const tensor =
        prepararImagemSFace(
            imageData
        );


    const nomeEntrada =
        sessaoSFace.inputNames[0];


    console.log(
        "11 - Executando SFace..."
    );


    const resultado =
        await sessaoSFace.run({

            [nomeEntrada]:
                tensor

        });


    const nomeSaida =
        sessaoSFace.outputNames[0];


    const embedding =
        resultado[
            nomeSaida
        ].data;


    console.log(
        "12 - Embedding SFace gerado."
    );


    console.log(
        "Quantidade de valores:",
        embedding.length
    );


    if (
        embedding.length !== 128
    ) {

        throw new Error(
            `SFace retornou ${embedding.length} valores em vez de 128.`
        );
    }


    return Array.from(
        embedding
    );
}
// ============================================================
// ENVIAR PARA FASTAPI
// ============================================================

async function enviarParaReconhecimento(
    embedding
) {

    console.log(
        "13 - Enviando biometria para o FastAPI..."
    );


    const resposta =
        await fetch(
            URL_FASTAPI,
            {
                method: "POST",

                headers: {

                    "Content-Type":
                        "application/json"

                },

                body:
                    JSON.stringify({

                        embedding:
                            embedding

                    })
            }
        );


    console.log(
        "STATUS HTTP:",
        resposta.status
    );


    if (!resposta.ok) {

        throw new Error(
            `FastAPI respondeu HTTP ${resposta.status}.`
        );
    }


    const resultado =
        await resposta.json();


    console.log(
        "14 - RESPOSTA DO FASTAPI:",
        resultado
    );


    return resultado;
}


// ============================================================
// MOSTRAR OPERADOR
// ============================================================

function mostrarOperador(
    usuario,
    similaridade,
    token
) {

    reconhecimentoConcluido =
        true;

    detectando =
        false;


    console.log(
        "================================"
    );

    console.log(
        "OPERADOR RECONHECIDO!"
    );

    console.log(
        "ID:",
        usuario.id
    );

    console.log(
        "NOME:",
        usuario.nome
    );

    console.log(
        "SIMILARIDADE:",
        similaridade
    );

    console.log(
        "================================"
    );


    atualizarStatus(
        `Operador identificado: ${usuario.nome}`,
        "sucesso"
    );


    if (operadorReconhecido) {

        operadorReconhecido.classList.remove(
            "oculto"
        );
    }


    if (nomeOperador) {

        nomeOperador.innerText =
            usuario.nome;
    }


    // Salva os dados do operador reconhecido
    sessionStorage.setItem(
        "kairosOperador",
        JSON.stringify({

            id:
                usuario.id,

            nome:
                usuario.nome

        })
    );


    // Salva o JWT REAL recebido da autenticação facial
    if (token) {

        sessionStorage.setItem(
            "kairosToken",
            token
        );

        console.log(
            "JWT da autenticação facial armazenado com sucesso."
        );

    } else {

        console.error(
            "ERRO: autenticação reconheceu o usuário, mas não recebeu JWT."
        );
    }


    // Mantém compatibilidade com o restante do sistema
    sessionStorage.setItem(
        "operador",
        usuario.nome
    );


    if (btnContinuar) {

        btnContinuar.classList.remove(
            "oculto"
        );

        btnContinuar.innerText =
            "Continuar";


        btnContinuar.onclick =
            () => {

                window.location.href =
                    "pacientes.html";

            };
    }


    setTimeout(
        () => {

            if (
                reconhecimentoConcluido
            ) {

                window.location.href =
                    "pacientes.html";
            }

        },
        1000
    );
}


// ============================================================
// PROCESSAR ROSTO
// ============================================================

async function processarRosto(
    resultadoYuNet,
    rostoDetectado
) {

    if (
        processando ||
        reconhecimentoConcluido
    ) {

        return;
    }


    processando =
        true;


    try {

        console.log(
            "9 - Rosto localizado. Gerando biometria..."
        );


        const deteccaoSofia =
            converterDeteccaoParaFormatoSofia(
                rostoDetectado
            );


        if (
            !deteccaoSofia
        ) {

            throw new Error(
                "Não foi possível converter a detecção facial."
            );
        }


        console.log(
            "DETECÇÃO NO FORMATO DA SOFIA:",
            deteccaoSofia
        );


        // ----------------------------------------------------
        // Obter ImageData original
        // ----------------------------------------------------

        const contexto =
            resultadoYuNet.canvas.getContext(
                "2d",
                {
                    willReadFrequently: true
                }
            );


        const imageData =
            contexto.getImageData(
                0,
                0,
                resultadoYuNet.canvas.width,
                resultadoYuNet.canvas.height
            );


        // ----------------------------------------------------
        // RECORTE EXATO DO CADASTRO DA SOFIA
        // ----------------------------------------------------

        const rostoRecortado =
            recortarRosto(
                imageData,
                deteccaoSofia
            );


        if (
            !rostoRecortado
        ) {

            throw new Error(
                "Não foi possível recortar o rosto."
            );
        }


        // ----------------------------------------------------
        // SFACE
        // ----------------------------------------------------

        const embedding =
            await gerarBiometriaSFace(
                rostoRecortado
            );


        console.log(
            "EMBEDDING SFACE - 128 VALORES:",
            embedding
        );


        // ----------------------------------------------------
        // BACKEND
        // ----------------------------------------------------

        const resultado =
            await enviarParaReconhecimento(
                embedding
            );


        if (
            resultado &&
            resultado.usuario
        ) {

            mostrarOperador(
                resultado.usuario,
                resultado.similaridade ||
                resultado.maiorSimilaridade,
                resultado.token
            );

        } else {

            console.log(
                "RESPOSTA DA AUTENTICAÇÃO:",
                resultado
            );


            if (
                resultado &&
                resultado.maiorSimilaridade !== undefined
            ) {

                atualizarStatus(

                    `Biometria comparada. Similaridade: ${resultado.maiorSimilaridade}`,

                    "aguardando"

                );

            } else {

                atualizarStatus(

                    "Biometria enviada para comparação.",

                    "aguardando"

                );
            }
        }


    } catch (erro) {

        console.error(
            "ERRO AO PROCESSAR ROSTO:",
            erro
        );


        atualizarStatus(
            "Erro durante o reconhecimento.",
            "erro"
        );

    } finally {

        processando =
            false;
    }
}
// ============================================================
// DETECÇÃO CONTÍNUA
// ============================================================

async function iniciarDeteccao() {

    if (detectando) {
        return;
    }


    detectando =
        true;


    console.log(
        "===== DETECÇÃO FACIAL INICIADA ====="
    );


    while (
        detectando &&
        !reconhecimentoConcluido
    ) {

        try {

            const agora =
                Date.now();


            if (
                agora -
                ultimoReconhecimento <
                INTERVALO_RECONHECIMENTO
            ) {

                await esperar(300);

                continue;
            }


            ultimoReconhecimento =
                agora;


            const resultado =
                await executarYuNet();


            if (!resultado) {

                await esperar(500);

                continue;
            }


            const deteccoes =
                interpretarYuNet(
                    resultado
                );


            const melhorRosto =
                selecionarMelhorRosto(
                    deteccoes
                );


            if (!melhorRosto) {

                atualizarStatus(
                    "Posicione seu rosto dentro da área.",
                    "aguardando"
                );

            } else {

                console.log(
                    "ROSTO DETECTADO:",
                    melhorRosto
                );


                atualizarStatus(
                    "Rosto detectado. Preparando identificação...",
                    "aguardando"
                );


                await processarRosto(
                    resultado,
                    melhorRosto
                );
            }


        } catch (erro) {

            console.error(
                "ERRO NA DETECÇÃO:",
                erro
            );


            atualizarStatus(
                "Erro na detecção facial.",
                "erro"
            );
        }


        await esperar(300);
    }
}


// ============================================================
// INICIALIZAÇÃO
// ============================================================

async function iniciarReconhecimento() {

    try {

        await iniciarCamera();

        await carregarModelosFaciais();


        if (
            sessaoYuNet &&
            sessaoSFace
        ) {

            await iniciarDeteccao();
        }

    } catch (erro) {

        console.error(
            "ERRO NA INICIALIZAÇÃO:",
            erro
        );
    }
}


// ============================================================
// INICIAR
// ============================================================

iniciarReconhecimento();