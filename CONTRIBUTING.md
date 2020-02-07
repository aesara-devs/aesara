If you want to contribute to Theano, have a look at the instructions here:
http://deeplearning.net/software/theano/dev_start_guide.html


## Migrating PRs from original Theano Repo
Theano-PyMC is actively merging new changes. If you have a pull request on the original respository and would like to move it here use the following commands in your local theano repo

```
# Go to your Theano Repo
cd /path/to/your/repo

# If you'd like to add theano-PyMC as a remote
git remote add pymc git@github.com:pymc-devs/Theano-PyMC.git

# Verify the changes. You should see the pymc-devs/Theano-PyMC.git
git remote -v

# Checkout the branch of your request
git checkout branch_name

# Push to Theano-PyMC
git push pymc branch_name
```

If you'd like to completely run this command instead

```
# If you'd like to replace this repo as a remote
git remote set-url origin git@github.com:pymc-devs/Theano-PyMC.git
```
