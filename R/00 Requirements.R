# ---- Requirements ----

package.list <- c("tidyverse",
                  "reshape2",
                  "caret",
                  "R.utils",
                  "viridis",
                  "Rtsne")

lapply(package.list, function(package) {
  if(!require(package, character.only=TRUE)) {
    install.packages(package)
    require(package, character.only=TRUE)
  }
})
