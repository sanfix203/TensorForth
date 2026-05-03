# Gabriele Sanfilippo SM3201618
#!/bin/bash

# Colori per l'output
GREEN='\033[0;32m'
RED='\033[0;31m'
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

echo "--- TEST POSITIVI ---"
# Stampa
run_test "Stampa multipla" "[ 5 ] p [ 10 ] p" 0 "data=[10]"

# Operazioni aritmetiche
run_test "Somma vettoriale" "[ 1 2 3 ] [ 10 20 30 ] + p" 0 "Tensor (shape=[3], data=[11 22 33])"
run_test "Sottrazione vettoriale" "[ 10 20 30 ] [ 1 2 3 ] - p" 0 "Tensor (shape=[3], data=[9 18 27])"
run_test "Moltiplicazione vettoriale" "[ 1 2 3 ] [ 4 5 6 ] * p" 0 "Tensor (shape=[3], data=[4 10 18])"

# Operazioni di comparazione
run_test "Maggiore" "[ 10 10 10 ] [ 5 10 15 ] > p" 0 "Tensor (shape=[3], data=[1 0 0])"
run_test "Uguale" "[ 10 10 10 ] [ 5 10 15 ] = p" 0 "Tensor (shape=[3], data=[0 1 0])"
run_test "Minore" "[ 10 10 10 ] [ 5 10 15 ] < p" 0 "Tensor (shape=[3], data=[0 0 1])"

# Operazioni logiche
run_test "And" "[ 0 1 0 1 ] [ 0 0 1 1 ] & p" 0 "Tensor (shape=[4], data=[0 0 0 1])"
run_test "Or" "[ 0 1 0 1 ] [ 0 0 1 1 ] | p" 0 "Tensor (shape=[4], data=[0 1 1 1])"
run_test "Not" "[ 0 1 ] ! p" 0 "Tensor (shape=[2], data=[1 0])"


echo -e "\n--- TEST NEGATIVI (Error Handling) ---"
run_test "Errore Sintassi: Tensore non chiuso" "[ 1 2 3 " 1 "Errore di sintassi"
run_test "Errore Sintassi: Spazio iniziale mancante" "[1 2 3 ]" 1 "Errore di sintassi"
run_test "Errore Sintassi: Spazio finale mancante" "[ 1 2 3]" 1 "Errore di sintassi"
run_test "Errore Sintassi: Separatore incorretto" "[ 1, 2, 3 ]" 1 "Errore di sintassi"
run_test "Errore Sintassi: Numero di spazi incorretto" "[1 2  3 ]" 1 "Errore di sintassi"
run_test "Errore Sintassi: Lettere nel tensore" "[ 1 A 3 ]" 1 "valore non numerico"
run_test "Errore Tipo: Somma con stringa" "[ 1 2 ] \"file.txt\" +" 1 "Errore di Tipo"
run_test "Errore Esecuzione: Dimensioni diverse" "[ 1 2 ] [ 1 2 3 ] +" 1 "shape diversa"
run_test "Errore Stack: Pop da stack vuoto" "+" 1 "Stack vuoto"
run_test "Errore Logico: Valori non booleani" "[ 1 2 0 ] [ 1 0 0 ] &" 1 "richiedono tensori composti solo da 0.0 e 1.0"

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