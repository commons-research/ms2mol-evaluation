#!/bin/bash
source ~/.bashrc

CURRENT_DIR=$(pwd) # because Sirius only accepts absolute paths
sirius \
    custom-db \
    create \
    --name=lotusISDB \
    --location=$CURRENT_DIR/data/sirius/lotusISDB.siriusdb ; 

sirius \
    custom-db \
    import \
    --db=lotusISDB \
    $CURRENT_DIR/data/sirius/sirius_custom_db.tsv ; 

