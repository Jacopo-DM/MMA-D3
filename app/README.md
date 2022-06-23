
# Downloading The Repo
To start with, we have to download this repository with the command:
```
git clone git@github.com:Jacopo-DM/MMA-D3.git
```
For the moment, before this branch gets merged with the `main` branch, you'll have to git-checkout the specific branch

```
git checkout -b svelte-testing
```

# Getting The Data
The app expects there to be a folder named `smallest` with `jpg` images, at the location:
```
/MMA-D3/app/static/smallest
```
This will be the data found in the zip file in our shared google drive called `smallest.zip`

To load a different folder of images, add the images to the `static` folder and change the following `index.svelte` line to match the change:

```
<Mesh imageFolder={'smallest'} imageExt={'jpg'} />
```
# Installing Node
The app runs using NodeJS, this can be downloaded and installed [here][1].

This will give you accesses to the node-package-manager (`npm` for short).

You can verify that `node` and `npm` are correctly installed on your system by running the following commands:

```bash
# Check the version of Node
node -v

# Check the version of npm
npm -v
```
## Mac OSX

Installation of NodeJS is OS-specific of course, the simplest way for mac users is to use [homebrew][2], using:

```bash
brew install node
```

# Developing

Once you've downloaded the project, and opened the `app` directory in your command line, you have to install the project dependencies with `npm install`

Finally, we can start a development server with:

```bash
npm run dev

# or start the server and open the app in a new browser tab
npm run dev -- --open
```

You can now open the website by going to `http://localhost:3000`

## Important Notes

Please disable ad-blockers and script-blockers when opening the local host, this may cause some javascript errors that make the program run slower

<!-- ## Building

To create a production version of your app:

```bash
npm run build
```

You can preview the production build with `npm run preview`.

> To deploy your app, you may need to install an [adapter](https://kit.svelte.dev/docs/adapters) for your target environment. -->

[1]: https://nodejs.org/en/about/
[2]: https://brew.sh/
