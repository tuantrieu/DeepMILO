## Requirement:
+ Dependency packages: keras 2.1.4, h5py, numpy
+ Homo_sapiens.GRCh37.75.dna.primary_assembly.fa (downloaded from ensembl) in the current folder

## Folder content:
* model: contains the loop model
* loopDB:
	* constitutive_loops.xlsx: list of constitutive loops
	* constitutive_loop_probability.txt: probability of constitutive loops without mutations 	
	
			


## Running
### Input: The program requires 2 input files:
  + a simple somatic mutation file from ICGC (in .tsv format)
  + a structural variant file from ICGC (in .tsv format)


### Output: 2 folders
  + one folder contain input files, one for each patient, contains sequences of loops taking into consideration mutations from patients
  + another folder contains output loop probability prediction for each patient

### Run: 
	python main.py ssm.tsv sv.tsv



