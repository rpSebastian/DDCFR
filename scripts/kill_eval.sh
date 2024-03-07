ps -ef | grep test_model | grep -v grep | awk '{print $2}' | xargs kill
ps -ef | grep run_ddcfr | grep -v grep | awk '{print $2}' | xargs kill
ps -ef | grep run_cfr | grep -v grep | awk '{print $2}' | xargs kill
