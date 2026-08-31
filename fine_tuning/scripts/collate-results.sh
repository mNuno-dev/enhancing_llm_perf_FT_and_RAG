#! /bin/bash

function get_task_score {
    task_name="$1"
    log_file="$2"
    if test -f "$log_file" && grep -q "$task_name" "$log_file"; then
        score=$(grep "$task_name" "$log_file" | cut -f 8 -d '|')
        printf %.2f $(bc <<< "$score * 100")
    else
        echo "na"
    fi
}

function get_task_score_from_following_line {
    task_name="$1"
    log_file="$2"
    if test -f "$log_file" && grep -q "$task_name" "$log_file"; then
        score=$(grep -A 1 "$task_name" "$log_file" | tail -n 1 | cut -f 8 -d '|')
        printf %.2f $(bc <<< "$score * 100")
    else
        echo "na"
    fi
}

function get_task_scores {
    log_file="$1"
    if test -f "$log_file"; then
        gpqa_diamond=$(get_task_score "Gervásio/META: GPQA-Diamond pt-PT (CoT, 0-shot)" "$log_file")
        mmlu=$(get_task_score "Gervásio/META: MMLU pt-PT (CoT, 0-shot)" "$log_file")
        mmlu_pro=$(get_task_score "Gervásio/META: MMLU-Pro pt-PT (CoT, 5-shot)" "$log_file")
        human_eval=$(get_task_score "Gervásio/META: HumanEval (0-shot, pass@1)" "$log_file")
        mgsm_flexible=$(get_task_score "Gervásio/META: MGSM pt-PT (CoT, 0-shot)" "$log_file")
        mgsm_strict=$(get_task_score_from_following_line "Gervásio/META: MGSM pt-PT (CoT, 0-shot)" "$log_file")
        mbpp_plus=$(get_task_score "Gervásio/META: MBPP Plus (0-shot, pass@1)" "$log_file")
        copa=$(get_task_score "Gervásio/ExtraGLUE: COPA pt-PT (generative, 0-shot)" "$log_file")
        mrpc=$(get_task_score "Gervásio/ExtraGLUE: MRPC pt-PT (generative, 0-shot)" "$log_file")
        rte=$(get_task_score "Gervásio/ExtraGLUE: RTE pt-PT (generative, 0-shot)" "$log_file")
        echo "$gpqa_diamond $mmlu $mmlu_pro $human_eval $mgsm_flexible $mgsm_strict $mbpp_plus $copa $mrpc $rte"
    else
        echo "gpqa_diamond mmlu mmlu_pro human_eval mgsm_flexible mgsm_strict mbpp_plus copa mrpc rte"
    fi |
    tr ' ' $'\t'
}

function collate_task_scores {
    model_size=$1
    log_dir=$2
    echo "model_size=$model_size  log_dir=$log_dir" >&2
    echo $'experiment\talpha\tseed\t'"$(get_task_scores "")"

    for log_file in $(find $log_dir -iname "eval-Llama-3.*-${model_size^^}-Instruct-seed*.log"); do
        model_name=$(grep -oP ".*(?=-seed[0-9]+)" <<< "$log_file")
        seed=$(grep -oP "(?<=$model_name-seed)[0-9]+" <<< "$log_file")
        echo $'baseline\tna\t'$seed$'\t'"$(get_task_scores "$log_file")"
    done

    for log_file in $(find $log_dir -iname "eval-gervasio-$model_size-*-alpha*-merged-seed*.log"); do
        experiment=$(grep -oP "(?<=gervasio-$model_size-)[0-9]+(?=-alpha[0-9]+-merged-seed[0-9]+)" <<< "$log_file")
        alpha=$(grep -oP "(?<=gervasio-$model_size-$experiment-alpha)[0-9]+(?=-merged-seed[0-9]+)" <<< "$log_file")
        seed=$(grep -oP "(?<=gervasio-$model_size-$experiment-alpha$alpha-merged-seed)[0-9]+" <<< "$log_file")
        echo "$experiment $alpha $seed $log_file"
    done | sort -nk1,1 -nk2,2 -nk3,3 | tr ' ' '-' |
    while read line; do
        experiment=$(cut -f 1 -d - <<< $line)
        alpha=$(cut -f 2 -d - <<< $line)
        seed=$(cut -f 3 -d - <<< $line)
        log_file=$(cut -f 4- -d - <<< $line)
        echo $experiment$'\t'$alpha$'\t'$seed$'\t'"$(get_task_scores $log_file)"
    done
}

if test $# != 2; then
    echo "usage: $0 MODELSIZE LOGDIR" >&2
    exit 2
fi

model_size=$1
log_dir=$2

collate_task_scores $model_size $log_dir
