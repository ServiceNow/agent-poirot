# Contributing to AgentPoirot

First off, thank you for considering contributing to AgentPoirot. It's people like you that make AgentPoirot such a great tool.

## Where do I go from here?

If you've noticed a bug or have a feature request, make sure to check our [Issues](https://github.com/ServiceNow/agent-poirot/issues) page to see if someone else in the community has already created a ticket. If not, go ahead and [make one](https://github.com/ServiceNow/agent-poirot/issues/new)!

## Fork & create a branch

If this is something you think you can fix, then fork AgentPoirot and create a branch with a descriptive name.

A good branch name would be (where issue #325 is the ticket you're working on):

```
git checkout -b 325-add-new-detection-algorithm
```

## Implement your fix or feature

At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner at first.

## Get the style right

Your patch should follow the same conventions & pass the same code quality checks as the rest of the project. We use:

- [Black](https://github.com/psf/black) for code formatting

We recommend you use VSCode with the Black extension to ensure your code is formatted correctly.
Otherwise, run `black .` in the root directory of the project to format your code.

## Make a Pull Request

At this point, you should switch back to your main branch and make sure it's up to date with AgentPoirot's main branch:

```
git remote add origin git@github.com:yourusername/AgentPoirot.git
git checkout main
git pull origin main
```

Then update your feature branch from your local copy of main, and push it!

```
git checkout 325-add-new-detection-algorithm
git rebase main
git push --set-origin origin 325-add-new-detection-algorithm
```

Finally, go to GitHub and [make a Pull Request](https://github.com/yourusername/AgentPoirot/compare) :D

## Keeping your Pull Request updated

If a maintainer asks you to "rebase" your PR, they're saying that a lot of code has changed, and that you need to update your branch so it's easier to merge.

To learn more about rebasing in Git, there are a lot of [good](https://git-scm.com/book/en/v2/Git-Branching-Rebasing) [resources](https://www.atlassian.com/git/tutorials/rewriting-history/git-rebase) but here's the suggested workflow:

```
git checkout 325-add-new-detection-algorithm
git pull --rebase origin main
git push --force-with-lease 325-add-new-detection-algorithm
```

## Merging a PR (maintainers only)

A PR can only be merged into main by a maintainer if:

* It is passing CI.
* It has no requested changes.
* It is up to date with current master.

## Thank You!

Your contributions to open source, large or small, make great projects like this possible. Thank you for taking the time to contribute to AgentPoirot.