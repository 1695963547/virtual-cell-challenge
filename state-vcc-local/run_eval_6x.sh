#\!/bin/bash
set -e
cd /home/zjh/state-vcc-local/state
rm -rf competition/vcc_test_eval/state_lg_6x
exec .venv/bin/python -c "
import logging,sys
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s: %(message)s')
sys.argv=['cell-eval','run',
  '-ap','competition/prediction_test_state_lg_6x.h5ad',
  '-ar','/home/zjh/vcc_official/adata_Test.h5ad',
  '-dr','competition/vcc_test_eval/baseline/real_de.csv',
  '-o','competition/vcc_test_eval/state_lg_6x',
  '--pert-col','target_gene',
  '--control-pert','non-targeting',
  '--profile','full',
  '--skip-metrics','pearson_edistance',
  '--num-threads','32']
from cell_eval.__main__ import main; main()
"
