import os 
import shutil

x  = os.listdir("IMFDB")
os.chdir("IMFDB")
#actor_names = []
for actors in x:
	print actors
	if(actors!='.DS_Store'):
		y = os.listdir(str(actors))
		os.chdir(str(actors))
		for films in y:
			if(films!='.DS_Store' and not films.endswith('.jpg')):
				os.chdir(str(films))
				z = os.listdir("images")
				os.chdir("images")
				print z
				for filename in z:
					shutil.copy(str(filename), "../..")
				os.chdir("..")
				os.chdir("..")
			if(films!='.DS_Store' and not films.endswith('.jpg')):
				shutil.rmtree(str(films))
		os.chdir("..")

