import os
#print(os.getcwd())
env_name = 'PYTHONPATH'
old = os.environ[env_name]
os.environ[env_name] = ':'+os.getcwd()+old
#print(os.environ[env_name])



import abs
print(dir(abs))

abs.main()
