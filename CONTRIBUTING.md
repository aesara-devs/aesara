If you want to contribute to Aesara, have a look at the instructions here:
http://deeplearning.net/software/aesara/dev_start_guide.html


## Migrating PRs from original Aesara Repo
Aesara-PyMC is actively merging new changes. If you have a pull request on the original respository and would like to move it here use the following commands in your local aesara repo

```
# Go to your Aesara Repo
cd /path/to/your/repo

# If you'd like to add aesara-PyMC as a remote
git remote add pymc git@github.com:pymc-devs/Aesara-PyMC.git

# Verify the changes. You should see the pymc-devs/Aesara-PyMC.git
git remote -v

# Checkout the branch of your request
git checkout branch_name

# Push to Aesara-PyMC
git push pymc branch_name
```

If you'd like to completely run this command instead

```
# If you'd like to replace this repo as a remote
git remote set-url origin git@github.com:pymc-devs/Aesara-PyMC.git
```
