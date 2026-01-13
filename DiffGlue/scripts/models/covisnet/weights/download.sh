pids=()

# gdown 1EEPrr1tECdn2oHk7gGIQIxSt-M_xGrqC & # 0kc5po4ee18_float32_jit_cpu_bev.ts
# pids+=($!)
gdown 1jkNRWmDNpwMTyeZ2MZtNVnsqFn3KQMlr & # 0kc5po4ee18_float32_jit_cpu_post.ts
pids+=($!)
gdown 10LkmMLUzmamch3n6oZ0dsxzEi3SvATFu & # 0kc5po4ee18_float32_jit_cpu_msg.ts
pids+=($!)
# gdown 1N0A7PzNSK2-N_Zim9RSLuHiJzgDpD-yU & # 0kc5po4ee18_float32_jit_cpu_bevdec.ts
# pids+=($!)
gdown 1-mUEsXj2zXD6NuWEM36E418_1YMnCvk6 & # 0kc5po4ee18_float32_jit_cpu_enc.ts
pids+=($!)

# gdown 1MqFyPKAkiXLbo6esDvVrqhRRqz16cxHv & # 0kc5po4ee18_float32_trt_post.ts
# pids+=($!)
# gdown 1coNgR7ZUE0prsubdQ0Uc7xbBNwxmjENS & # 0kc5po4ee18_float32_trt_msg.ts
# pids+=($!)
# gdown 1_10PLLkWCnFUdwl9IOYuzMk5X7dbWC0k & # 0kc5po4ee18_float32_trt_enc.ts
# pids+=($!)
# gdown 1FSp1XNoRKy-wMploKpM7v4BLTcAinxuP & # 0kc5po4ee18_float32_trt_bevdec.ts
# pids+=($!)
# gdown 17DYxLqO0JAhXCM6UGknTuHdvXzIOTdfy & # 0kc5po4ee18_float32_trt_bev.ts
# pids+=($!)

wait "${pids[@]}" # Wait for all processes
echo "All done!"