const pacientes = [
    {
        id: 1,
        nome: "Paciente de teste 01",
        exame: "Hemograma"
    },
    {
        id: 2,
        nome: "Paciente de teste 02",
        exame: "Análise de urina"
    }
];


const lista = document.getElementById("lista-pacientes");

const nomeOperador =
    document.getElementById("nome-operador");


// Recupera o operador que fez o reconhecimento facial
const operadorSalvo =
    sessionStorage.getItem("kairos_operador");


if (operadorSalvo) {

    nomeOperador.innerText = operadorSalvo;

} else {

    nomeOperador.innerText = "Operador não identificado";

}


// Cria os cartões dos pacientes
pacientes.forEach((paciente) => {

    const botao = document.createElement("button");

    botao.className = "paciente";

    botao.innerHTML = `
        <span class="paciente-nome">
            ${paciente.nome}
        </span>

        <span class="paciente-exame">
            Exame: ${paciente.exame}
        </span>
    `;


    botao.addEventListener("click", () => {

        // Guarda os dados do paciente
        sessionStorage.setItem(
            "kairos_paciente_id",
            paciente.id
        );

        sessionStorage.setItem(
            "kairos_paciente_nome",
            paciente.nome
        );

        sessionStorage.setItem(
            "kairos_exame",
            paciente.exame
        );


        // Vai para o monitoramento
        window.location.href = "kairos.html";

    });


    lista.appendChild(botao);

});