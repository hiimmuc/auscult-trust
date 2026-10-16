15/10/2026 17:42 | A 0/149 (sweep alive, 1 OOM requeue job 3, self-recovered) | B 3/201 (alive, no failures) | NaN none | disk A 141G B 402G | normal
15/10/2026 17:51 | A 1/149 (sweep alive, pid changed 1472800->1757654, restarted outside monitor?) | B 3/201 (alive, pid changed 757228->770321) | NaN none | FAILED 0 | disk A 140G B 402G | odd: both sweep pids changed, no action taken
15/10/2026 17:57 | A 1/149 (alive, pid 1757654) | B 3/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 140G B 402G | normal; handoff re-read (OOM now reportable)
15/10/2026 20:19 | A 4/149 (alive, pid 1757654) | B 9/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 148G B 402G | normal
15/10/2026 21:20 | A 5/149 (alive, pid 1757654) | B 13/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 148G B 400G | normal
15/10/2026 22:20 | A 6/149 (alive, pid 1757654) | B 15/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 148G B 399G | normal; behind 60/machine target by 17/10
15/10/2026 23:20 | A 7/149 (alive, pid 1757654) | B 16/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 146G B 398G | normal; slow vs 60/machine by 17/10
16/10/2026 00:20 | A 8/149 (alive, pid 1757654) | B 18/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 146G B 397G | normal; slow vs 60/machine by 17/10
16/10/2026 01:20 | A 10/149 (alive, pid 1757654) | B 22/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 144G B 397G | normal; slow vs 60/machine by 17/10
16/10/2026 02:19 | A 10/149 (alive, pid 1757654, no change since 01:19) | B 25/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 144G B 397G | normal; slow vs 60/machine by 17/10
16/10/2026 03:19 | A 11/149 (alive, pid 1757654) | B 30/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 144G B 396G | normal; slow vs 60/machine by 17/10
16/10/2026 04:20 | A 14/149 (alive, pid 1757654) | B 32/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 144G B 395G | normal; slow vs 60/machine by 17/10
16/10/2026 05:20 | A 14/149 (alive, pid 1757654, no change since 04:19) | B 38/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 143G B 394G | normal; slow vs 60/machine by 17/10
16/10/2026 06:20 | A 17/149 (alive, pid 1757654) | B 46/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 143G B 394G | normal; A behind 60/machine target by 17/10
16/10/2026 07:20 | A 19/149 (alive, pid 1757654) | B 50/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 144G B 392G | normal; A behind 60/machine target by 17/10
16/10/2026 08:19 | A 22/149 (alive, pid 1757654) | B 51/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 143G B 393G | normal; A behind 60/machine target by 17/10
16/10/2026 09:19 | A 26/149 (alive, pid 1757654) | B 54/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 143G B 393G | normal; A behind 60/machine target by 17/10
16/10/2026 10:19 | A 30/149 (alive, pid 1757654) | B 55/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 140G B 391G | normal; A behind 60/machine target by 17/10
16/10/2026 11:19 | A 30/149 (alive, pid 1757654, no change since 10:19) | B 58/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 140G B 388G | normal; A behind 60/machine target by 17/10
16/10/2026 12:20 | A 30/149 (alive, pid 1757654, flat since 10:19; running beats+ft+patchmix_cl) | B 60/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 140G B 389G | A slow: 60/machine target by 17/10 at risk
16/10/2026 13:20 | A 32/149 (alive, pid 1757654) | B 62/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | disk A 139G B 387G | normal; A behind 60/machine target by 17/10 (A now moving)
16/10/2026 14:20 | A 30/120 (main sweep pid 1757654 NOT RUNNING; list now 120 lines; competing sweep pid 4086365 grid-v1-sgscl-test.txt active; restart HELD) | B 61/201 (alive, pid 770321) | NaN none | FAILED 0 | OOM 0 | collapsed: A/33.log (Terminated), B/53.log | disk A 136G B 388G | report to user
16/10/2026 15:19 | A 30/120 (sweep pid 4190772 alive, restarted by someone else; grid-v1-A/ log dir missing, status script error) | B 64/230 (pid 958673, list now 230 not 201) | NaN none | FAILED 0 | OOM 0 | collapsed none | disk A 135G B 387G | report to user
16/10/2026 16:20 | A 30/120 (alive, pid 4190772, flat since 15:19; per-job log dir missing, status script error) | B 67/230 (alive, pid 958673) | NaN none | FAILED 0 | OOM 0 | collapsed none | disk A 135G B 387G | report to user
16/10/2026 17:20 | A 32/120 (alive, pid 4190772) | B 69/230 (alive, pid 958673) | NaN none | FAILED 0 | OOM 0 | collapsed none | disk A 132G B 385G | normal; A log dir still missing (status script error)
16/10/2026 18:14 | A 32/120 (alive, pid 4190772, flat since 17:19) | B 70/230 (alive, pid 958673) | NaN none | FAILED 0 | OOM 0 | collapsed none | disk A 135G B 385G | normal; A log dir still missing (status script error); A flat 1h
16/10/2026 18:30 | correction (main session): logs/grid-v1-A/ exists since 15:00; it was empty until the first job started at 16:35, so the collapse loop hit a literal '*.log'. grid_status.sh now skips that case. A per-job NaN/collapse checks work; A jobs 3/33/34 at epochs 24-32 of 50, so 'A flat 1h' is normal.
