##Requirement:
+ Dependent packages: keras 2.1.4, h5py, number
+ Homo_sapiens.GRCh37.75.dna.primary_assembly.fa in the current folder

##Running
###Input: The program requires 2 input files:
  + a simple somatic mutation file from ICGC (in .tsv format)
  + a structural variant file from ICGC (in .tsv format)


###Output: 2 folders
  + one folder contain input files, one for each patient, contains sequences of loops taking into consideration mutations from patients
  + another folder contains output loop probability prediction for each patient

###Run: 
	python main.py ssm.tsv sv.tsv



