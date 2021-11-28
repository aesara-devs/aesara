If you want to contribute to Aesara, have a look at the instructions here:
https://aesara.readthedocs.io/en/latest/dev_start_guide.html


## Migrating PRs from the original Theano Repo
Aesara is actively merging new changes. If you have a pull request on the original Theano repository and would like to move it here use the following commands in your local Aesara repository:

```
# Go to your Aesara repo
cd /path/to/your/repo

# If you'd like to add aesara as a remote
git remote add aesara git@github.com:aesara-devs/aesara.git

# Verify the changes. You should see the aesara-devs/aesara.git
git remote -v

# Checkout the branch of your request
git checkout branch_name

# Push to Aesara
git push aesara branch_name
```

If you would like to make Aesara the new "main" upstream remote:

```
git remote set-url upstream git@github.com:aesara-devs/aesara.git
```
