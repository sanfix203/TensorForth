# Gabriele Sanfilippo SM3201618
#!/bin/bash

# Colori per l'output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[1;34m'
ORANGE='\033[0;33m'
NC='\033[0m' # Nessun colore
EXEC="./tensorforth"

PASSED=0
FAILED=0

# Compilazione
echo "Compilazione in corso..."
make > /dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Errore di compilazione. Test annullati.${NC}"
    exit 1
fi
echo -e "Compilazione completata. Inizio Test...\n"

# Funzione per eseguire un singolo test
# Uso: run_test "Nome Test" "Contenuto del file" "Exit code atteso (0=ok, 1=errore)" "Testo atteso nell'output"
run_test() {
    TEST_NAME=$1
    TF_CONTENT=$2
    EXPECTED_EXIT_CODE=$3
    EXPECTED_OUTPUT=$4

    echo "$TF_CONTENT" > temp_test.tf

    # Esegue il programma catturando sia l'output normale che gli errori
    OUTPUT=$($EXEC temp_test.tf 2>&1)
    ACTUAL_EXIT_CODE=$?

    if [ $ACTUAL_EXIT_CODE -ne $EXPECTED_EXIT_CODE ]; then
        echo -e "${RED}[FAIL]${NC} $TEST_NAME"
        echo "       Atteso exit code $EXPECTED_EXIT_CODE, ottenuto $ACTUAL_EXIT_CODE."
        echo "       Output generato: $OUTPUT"
        FAILED=$((FAILED+1))
        rm temp_test.tf
        return
    fi

    # Verifica se l'output contiene il testo atteso
    if echo "$OUTPUT" | grep -Fiq "$EXPECTED_OUTPUT"; then
        echo -e "${GREEN}[OK]${NC}   $TEST_NAME"
        PASSED=$((PASSED+1))
    else
        echo -e "${RED}[FAIL]${NC} $TEST_NAME"
        echo "       Output atteso contenente: '$EXPECTED_OUTPUT'"
        echo "       Output ottenuto: '$OUTPUT'"
        FAILED=$((FAILED+1))
    fi

    rm temp_test.tf
}


printf "${COLOR_BLUE}--- TEST POSITIVI ---${NC}\n"
printf "${BLUE}Stampa${NC}\n"
run_test "Stampa multipla" "[ 5 ] p [ 10 ] p" 0 "data=[10]"

printf "\n${BLUE}Operazioni aritmetiche${NC}\n"
run_test "Somma vettoriale" "[ 1 2 3 ] [ 10 20 30 ] + p" 0 "Tensor (shape=[3], data=[11 22 33])"
run_test "Sottrazione vettoriale" "[ 10 20 30 ] [ 1 2 3 ] - p" 0 "Tensor (shape=[3], data=[9 18 27])"
run_test "Moltiplicazione vettoriale" "[ 1 2 3 ] [ 4 5 6 ] * p" 0 "Tensor (shape=[3], data=[4 10 18])"

printf "\n${BLUE}Operazioni di comparazione${NC}\n"
run_test "Maggiore" "[ 10 10 10 ] [ 5 10 15 ] > p" 0 "Tensor (shape=[3], data=[0 0 1])"
run_test "Uguale" "[ 10 10 10 ] [ 5 10 15 ] = p" 0 "Tensor (shape=[3], data=[0 1 0])"
run_test "Minore" "[ 10 10 10 ] [ 5 10 15 ] < p" 0 "Tensor (shape=[3], data=[1 0 0])"

printf "\n${BLUE}Operazioni logiche${NC}\n"
run_test "And" "[ 0 1 0 1 ] [ 0 0 1 1 ] & p" 0 "Tensor (shape=[4], data=[0 0 0 1])"
run_test "Or" "[ 0 1 0 1 ] [ 0 0 1 1 ] | p" 0 "Tensor (shape=[4], data=[0 1 1 1])"
run_test "Not" "[ 0 1 ] ! p" 0 "Tensor (shape=[2], data=[1 0])"

printf "\n${BLUE}Operazioni di selezione${NC}\n"
run_test "Selezione" "[ 5 6 7 8 ] [ 1 2 3 4 ] [ 1 1 0 0 ] $ p" 0 "Tensor (shape=[4], data=[1 2 7 8])"

printf "\n${BLUE}Operazioni specifiche per tensori${NC}\n"
run_test "Prodotto matrici 2D @ 2D" "[ 2 0 1 2 ] [ 2 2 ] r [ 1 2 3 4 ] [ 2 2 ] r @ p" 0 "data=[4 4 10 8]"
run_test "Prodotto matrici 1D @ 2D" "[ 1 2 3 4 5 6 ] [ 2 3 ] r [ 1 2 ] @ p" 0 "data=[9 12 15]"
run_test "Prodotto matrici 2D @ 1D" "[ 5 6 ] [ 1 2 3 4 ] [ 2 2 ] r @ p" 0 "data=[17 39]"
run_test "Prodotto scalare" "[ 1 2 3 ] [ 0 1 2 ] . p" 0 "Tensor (shape=[1], data=[8])"

printf "\n${BLUE}Operazioni sulla forma dei tensori${NC}\n"
run_test "Reshape" "[ 1 2 3 4 ] [ 2 2 ] r p" 0 "Tensor (shape=[2 2], data=[1 2 3 4])"
run_test "Ravel" "[ 1 2 3 4 ] [ 2 2 ] r _ p" 0 "Tensor (shape=[4], data=[1 2 3 4])"
run_test "Shape 1D" "[ 1 2 3 4 ] # p" 0 "Tensor (shape=[1], data=[4])"
run_test "Shape 2D" "[ 1 2 3 4 5 6 ] [ 3 2 ] r # p" 0 "Tensor (shape=[2], data=[3 2])"

printf "\n${BLUE}Operazioni elemento per elemento${NC}\n"
run_test "Relu" "[ -1 0 1 ] R p" 0 "Tensor (shape=[3], data=[0 0 1])"
run_test "Min" "[ -1 0 1 ] [ 1 0 -1 ] m p" 0 "Tensor (shape=[3], data=[-1 0 -1])"
run_test "Max" "[ -1 0 1 ] [ 1 0 -1 ] M p" 0 "Tensor (shape=[3], data=[1 0 1])"

printf "\n${BLUE}Operazioni di riduzione${NC}\n"
run_test "Sommatoria" "[ -1 2 3 ] S p" 0 "Tensor (shape=[1], data=[4])"

printf "\n${BLUE}Operazioni di filling di tensori${NC}\n"
run_test "Fill 2D ciclico" "[ 2 3 ] [ 1 2 ] f p" 0 "data=[1 2 1 2 1 2]"
run_test "Fill 1D singolo" "[ 5 ] [ 9 ] f p" 0 "shape=[5], data=[9 9 9 9 9]"

printf "\n\n${ORANGE}--- TEST NEGATIVI ---${NC}\n"
printf "${ORANGE}Errori di Stack (Underflow)${NC}\n"
run_test "Stack vuoto per Somma" "[ 1 2 3 ] +" 1 "vuoto"
run_test "Stack vuoto per Stampa" "p" 1 "vuoto"
run_test "Stack vuoto per Unario" "R" 1 "vuoto"

printf "\n${ORANGE}Errori di Forma (Shape Mismatch)${NC}\n"
run_test "Somma lunghezze diverse" "[ 1 2 ] [ 1 2 3 ] +" 1 "shape diversa"
run_test "Confronto 1D e 2D" "[ 1 2 3 4 ] [ 1 2 3 4 ] [ 2 2 ] r =" 1 "shape diversa"
run_test "Selezione maschera errata" "[ 1 2 ] [ 3 4 ] [ 1 0 1 ] $" 1 "shape diversa"

printf "\n${ORANGE}Errori Logici${NC}\n"
run_test "Valori non booleani" "[ 1 2 0 ] [ 1 0 0 ] &" 1 "composti"

printf "\n${ORANGE}Errori Operazioni Reshape${NC}\n"
run_test "Reshape volume errato" "[ 1 2 3 4 ] [ 3 3 ] r" 1 "numero di elementi"
run_test "Reshape forma errata" "[ 1 2 3 4 ] [ 2 2 1 ] r" 1 "tensore forma"
run_test "Dot product incompatibile" " [ 1 2 3 ] [ 1 2 ] ." 1 "shape diversa"
run_test "Matmul dimensioni incompatibili" "[ 1 2 3 4 5 6 ] [ 3 2 ] r [ 1 2 3 4 ] [ 2 2 ] r @" 1 "Dimensioni interne"

printf "${ORANGE}Errori di filling${NC}\n"
run_test "Fill con s non 1D" "[ 1 2 3 4 ] [ 2 2 ] r [ 1 2 ] f" 1 "1D"
run_test "Fill con s troppo lungo" "[ 2 2 2 ] [ 1 ] f" 1 "forma"
run_test "Fill con dimensione zero" "[ 0 5 ] [ 1 ] f" 1 "dimensioni"

echo -e "\n--- RISULTATI ---"
echo -e "Passati: ${GREEN}$PASSED${NC}"
echo -e "Falliti: ${RED}$FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}Tutti i test sono passati.${NC}"
    exit 0
else
    echo -e "${RED}Alcuni test sono falliti. Controlla il codice.${NC}"
    exit 1
fi