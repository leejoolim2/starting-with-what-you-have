suppressMessages(library(QCA))
d <- read.csv("/home/claude/repo/starting-with-what-you-have/ch06-fsqca/data/upgrading_cities.csv")
CONDS <- c("tenure","finance","participation","capacity")

# same calibration anchors as the book: full_in 8, crossover 5, full_out 2
cal <- d
for (v in c(CONDS,"upgrade_success")) cal[[v]] <- calibrate(d[[v]], type="fuzzy", thresholds=c(2,5,8))

cat("=== calibration (first 6 rows) ===\n")
print(round(head(cal[,c(CONDS,"upgrade_success")],6),4))

cat("\n=== necessity ===\n")
ns <- superSubset(cal, outcome="upgrade_success", conditions=CONDS, relation="nec", incl.cut=0.90)
print(ns)

cat("\n=== truth table ===\n")
tt <- truthTable(cal, outcome="upgrade_success", conditions=CONDS, incl.cut=0.80, n.cut=1, show.cases=TRUE)
print(tt)

cat("\n=== parsimonious solution ===\n")
sp <- minimize(tt, include="?", details=TRUE)
print(sp)

cat("\n=== complex solution ===\n")
sc <- minimize(tt, details=TRUE)
print(sc)
