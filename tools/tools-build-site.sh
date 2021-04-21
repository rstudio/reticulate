
set -eux

: "${TMPDIR:=/tmp}"

if [ -z "${VERSION}" ]; then
  echo "Usage: VERSION=<version> make site"
  exit 0
fi

cd "${TMPDIR}"

rm -rf reticulate-deploy
mkdir reticulate-deploy
cd reticulate-deploy

git clone -b "${VERSION}" https://github.com/rstudio/reticulate reticulate
git clone -b gh-pages https://github.com/rstudio/reticulate site

cd reticulate
R -s -e 'pkgdown::build_site()'
cd ..

cd site
rm -rf *
cd ..

cp -R reticulate/docs/ site/

cd site
rm reference/Rplot*
git add -A
git commit -m "Build site for reticulate: ${VERSION}"
git push -u
cd ..

cd ..
rm -rf reticulate-deploy
