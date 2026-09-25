const URL_FASTAPI = "http://127.0.0.1:8000";

const lista = document.getElementById("lista-pacientes");
const nomeOperador = document.getElementById("nome-operador");

// Declaração da função como ASYNC para permitir o uso do await
async function carregarPacientes() {
    // 1. Recupera o operador do sessionStorage
    const operadorSalvo = sessionStorage.getItem("kairosOperador") || sessionStorage.getItem("kairos_operador");
    
    // Recupera o token salvo na sessão após o reconhecimento facial
    const token = sessionStorage.getItem("kairosToken");
console.log(
    "[pacientes.js] JWT disponível:",
    token ? "SIM" : "NÃO"
);

    let usuarioId = "";
    let nomeExibicao = "Operador não identificado";

    if (operadorSalvo) {
        try {
            const operador = JSON.parse(operadorSalvo);
            usuarioId = operador.id || "";
            nomeExibicao = operador.nome || "Operador";
        } catch (e) {
            usuarioId = operadorSalvo;
            nomeExibicao = operadorSalvo;
        }
    }

    if (nomeOperador) {
        nomeOperador.innerText = nomeExibicao;
    }

    console.log(`[pacientes.js] Operador ativo: ${nomeExibicao} | ID: '${usuarioId}'`);

    try {
        // Timestamp para evitar cache do navegador
        const timestamp = new Date().getTime();
        const urlFinal = `${URL_FASTAPI}/pacientes?usuario_id=${usuarioId}&t=${timestamp}`;

        console.log(`[pacientes.js] Requisitando sem cache: ${urlFinal}`);

        // O await está corretamente dentro da função async com os headers de autorização
        const resposta = await fetch(urlFinal, { 
            method: "GET",
            headers: {
                "Content-Type": "application/json",
                "Authorization": token ? `Bearer ${token}` : ""
            },
            cache: "no-store" 
        });

        if (!resposta.ok) {
            throw new Error(`Erro HTTP ${resposta.status}`);
        }

        const pacientes = await resposta.json();
        console.log("[pacientes.js] Resposta do backend:", pacientes);

        if (!lista) {
            console.error("[pacientes.js] Elemento #lista-pacientes não encontrado no HTML!");
            return;
        }

        lista.innerHTML = "";

        if (!Array.isArray(pacientes) || pacientes.length === 0) {
            lista.innerHTML = "<p class='sem-pacientes'>Nenhum paciente cadastrado para este operador.</p>";
            return;
        }

        // Renderização dos botões dos pacientes
        pacientes.forEach((paciente) => {
            const botao = document.createElement("button");
            botao.className = "paciente";

            const exameNome = paciente.exame || paciente.tipo_exame || paciente.observacoes || "Análise Laboratorial";

            botao.innerHTML = `
                <span class="paciente-nome">
                    ${paciente.nome}
                </span>

                <span class="paciente-exame">
                    Exame: ${exameNome}
                </span>
            `;

            botao.addEventListener("click", () => {
                sessionStorage.setItem("kairos_paciente_id", paciente.id);
                sessionStorage.setItem("kairos_paciente_nome", paciente.nome);
                sessionStorage.setItem("kairos_exame", exameNome);

                window.location.href = "kairos.html";
            });

            lista.appendChild(botao);
        });

    } catch (erro) {
        console.error("[pacientes.js] Erro ao carregar pacientes:", erro);
        if (lista) {
            lista.innerHTML = "<p class='erro'>Erro ao carregar a lista de pacientes.</p>";
        }
    }
}

// Executa a função async após o carregamento da página
document.addEventListener("DOMContentLoaded", () => {
    carregarPacientes();
});