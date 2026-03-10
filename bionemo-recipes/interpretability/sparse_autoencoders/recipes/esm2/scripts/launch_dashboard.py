from esm2_sae import launch_protein_dashboard

proc = launch_protein_dashboard('./dash/features_atlas.parquet', features_dir='dash')
input('Dashboard running. Press enter to stop. \n')
proc.terminate()
