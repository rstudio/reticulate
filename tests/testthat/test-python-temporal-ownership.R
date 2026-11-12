context("temporal ownership")

test_that("datetime conversion releases references to iterated scalars", {
  skip_if_no_pandas()
  objects <- py_run_string("
def make_fixture():
    import datetime
    import sys
    import pandas as pd

    dates = (datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 2))

    class DateSeries(pd.Series):
        def __iter__(self):
            return iter(dates)

    class DateFrame(pd.DataFrame):
        _constructor_sliced = DateSeries

    def refcounts():
        return [sys.getrefcount(value) for value in dates]

    frame = DateFrame({'x': pd.to_datetime(['2020-01-01', '2020-01-02'])})
    return {'frame': frame, 'refcounts': refcounts}

fixture = make_fixture()
", local = TRUE, convert = FALSE)
  frame <- objects$fixture$frame
  refcounts <- objects$fixture$refcounts
  before <- py_to_r(refcounts())
  for (i in seq_len(3L)) {
    actual <- py_to_r(frame)
    expect_identical(as.numeric(actual$x), c(1577836800, 1577923200))
    expect_identical(py_to_r(refcounts()), before)
  }
})
