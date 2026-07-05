let streams = {};
let cameras = [];

async function listarCameras() {
    try {
        await navigator.mediaDevices.getUserMedia({ video: true });

        const dispositivos = await navigator.mediaDevices.enumerateDevices();

        cameras = dispositivos.filter(device => device.kind === "videoinput");

        console.log("Câmeras encontradas:", cameras);

        for (let i = 0; i < cameras.length && i < 4; i++) {
            ligarCamera(i + 1);
        }

    } catch (erro) {
        alert("Erro ao listar câmeras: " + erro);
    }
}

async function ligarCamera(numero) {
    const video = document.getElementById(`camera${numero}`);

    if (!cameras[numero - 1]) {
        alert(`Câmera ${numero} não encontrada.`);
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                deviceId: {
                    exact: cameras[numero - 1].deviceId
                }
            },
            audio: false
        });

        streams[numero] = stream;
        video.srcObject = stream;

    } catch (erro) {
        alert("Erro ao ligar câmera " + numero + ": " + erro);
    }
}

function desligarCamera(numero) {
    if (streams[numero]) {
        streams[numero].getTracks().forEach(track => track.stop());
        document.getElementById(`camera${numero}`).srcObject = null;
        delete streams[numero];
    }
}

function expandir(bloco) {
    bloco.classList.add("expandido");
}

function voltar(event, botao) {
    event.stopPropagation();
    botao.parentElement.classList.remove("expandido");
}

// ← NOVO: fecha com ESC
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        const expandido = document.querySelector('.expandido');
        if (expandido) {
            expandido.classList.remove('expandido');
        }
    }
});

window.onload = listarCameras;