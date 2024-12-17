mkdir -p designed
for pdb in structure_predictions/*.pdb; do
    pdb_basename=$(basename "$pdb" .pdb)
    pdb_selchain -A "$pdb" > "designed/${pdb_basename}_chainA.pdb"
done
