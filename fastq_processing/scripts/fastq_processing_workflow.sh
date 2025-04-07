#!/bin/bash

#inputs
#the paired end files should be named as ${NAME}_R{1,2}_001.fastq and
#stored in the fastq/ folder
URNAME=
       
#set options
DO_QC=true
DO_TRIM=true
DO_JOIN=true
DO_CAT=true
DO_LEN=true
PAUSE=false

module load fastqc
module load trimmomatic/0.36
SCRIPT_PTH=/n/groups/marks/projects/nanobodies/kruse_nanobody_sequencing/scripts
export PATH=$SCRIPT_PTH:$PATH


for NAME in ### fill in
do
    #intial QC
    QCRES="fastqc/${NAME}/"
    mkdir -p ${QCRES}
    if [ ${DO_QC} = true ] ; then
        echo "QC for ${NAME}"; echo
        fastqc -o ${QCRES} fastq/${NAME}_R1_001.fastq fastq/${NAME}_R2_001.fastq 
    
        if [ ${PAUSE} = true ] ; then
            echo "Pause to check the QC results"
            echo "Hit enter to continue..."; echo
            read stop1
        fi
    fi

    #trimmomatic
    FILDIR="filtered_fastq"
    mkdir -p ${FILDIR}
    # FastQC didnt show adapters, so will not remove them.
    # Not removing leading sequences, since our barcodes are there, and this could introduce novel barcodes. Better to just let the sequence be filtered out at barcode splitting
    # Not removing low quality trailing sequences since we are joining pairs
    # Will be filtering for sequence length after joining reads, so need to specify a minlen
    # Since we are looking only for full pairs, just throw out pairs where one read survived and the partner didnt\
    if [ ${DO_TRIM} = true ] ; then
        echo "Trimmomatic for ${NAME}"; echo
        java -jar $TRIMMOMATIC/trimmomatic-0.36.jar PE -phred33 fastq/${NAME}_R1_001.fastq fastq/${NAME}_R2_001.fastq \
                                                    ${FILDIR}/${NAME}_R1_001_filtered.fastq /dev/null \
                                                    ${FILDIR}/${NAME}_R2_001_filtered.fastq /dev/null SLIDINGWINDOW:4:15
    fi

    #pair end joining
    if [ ${DO_JOIN} = true ] ; then
        echo "Pair end joining for ${NAME}"; echo
        joinstart=`date +%s`
        fastq-join -v ' ' ${FILDIR}/${NAME}_R1_001_filtered.fastq ${FILDIR}/${NAME}_R2_001_filtered.fastq -o ${FILDIR}/${NAME}_%.fastq
        joinend=`date +%s`
        jointime=$((joinend-joinstart)); echo Runtime: ${jointime}s
        
        if [ ${PAUSE} = true ] ; then
            echo "Pause to check the pair end joining results"
            echo "Hit enter to continue..."; echo
            read stop2
        fi
        
        #delete the unpaired reads
        rm ${FILDIR}/*_un1.fastq
        rm ${FILDIR}/*_un2.fastq
    fi

    
done

#concatenate all the results
if [ ${DO_CAT} = true ] ; then
    echo "Concatenating all join results"; echo
    cat ${FILDIR}/*_join.fastq > ${FILDIR}/${URNAME}_alljoin.fastq
fi


#filter joined reads by length
if [ ${DO_LEN} = true ] ; then
    echo "Filtering by length"; echo
    #first get the lengths with fastqc
    fastqc -o ${QCRES} ${FILDIR}/${URNAME}_alljoin.fastq
    unzip ${QCRES}/${URNAME}_alljoin_fastqc.zip -d ${QCRES}
    cat ${QCRES}/${URNAME}_alljoin_fastqc/fastqc_data.txt | grep "Sequence Length Distribution" -A 100 | sed '/>>END_MODULE/q'
    
    echo "Enter the min-length for filtering"
    read MIN
    echo "min-length used: ${MIN}"
    echo "Enter the max-length for filtering"
    read MAX
    echo "max-length used: ${MAX}"
    
    #filter
    python $SCRIPT_PTH/fastq_length_filter.py -i ${FILDIR}/${URNAME}_alljoin.fastq -o ${FILDIR}/${URNAME}_lenfilter.fastq --min-len $MIN --max-len $MAX
fi









