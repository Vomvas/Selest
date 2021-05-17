""" Class for printing out tables """


__all__ = ["Table", "LatexTable", "ACMTable"]


class Cell(object):
    """
    Helper class for table cell formatting. Initialized by the table it belongs to and the position in the table.

    t: Table object
    pos: int tuple denoting the position in the table (0-indexed)

    Example:
    t = Table(3,3)
    pos = (1,2)
    c = Cell(t, pos)        # Denotes the cell at row 2 and column 3.
    """

    def __init__(self, table, pos):
        self.table = table
        self.pos = pos


class Table(object):
    """ Base class """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols


class ACMTable(object):
    """
    Helper class to output lines for tables in the ACM format to be copied in the paper
    """

    def __init__(self, t_type="runtimes_costs"):

        self.col_separator = ' & '
        self.line_separator = "\\\\"
        self.t_type = t_type


class LatexTable(object):
    """
    Helper class representing a Table object formatted for LaTeX. Current implementation is restricted to the format
    used in the Secure Music write-up to display the execution runtimes
    """

    @staticmethod
    def format_from_file(filename):
        """ Loads the format of the sec_music runtimes table from file """
        with open(filename) as f:
            return f.read()

    def __init__(self, format_file):
        self.table = LatexTable.format_from_file(format_file)
        self.multirow_separator = "\\hline\n            \\hline\n"
        self.col_separator = '&'
        self.n_parties_separator = "\\\\"
        self.column_labels = ["protocol", "n_players", "sec_music", "vector_projection", "kernel_approx"]
        self.row_group_labels = ["pre", "mascot (online)", "mascot (combined)", "mal-shamir (online)",
                                 "mal-shamir (combined)", "yao", "post"]

        self.runtime_pattern = [" \\textcolor{\\completedColor}{", "runtime", "} "]

    @property
    def row_groups(self):
        """ Return row groups """
        return self.table.split(self.multirow_separator)

    def fill_runtime_pattern(self, x, col, val):
        """ Fill the value in the runtime pattern """
        if not val:
            val = " - "
        new_res = x.split(self.col_separator)
        new_res[col] = "".join(x if x != "runtime" else str(val) for x in self.runtime_pattern)
        return self.col_separator.join(new_res)

    def line_for_n_parties(self, line, n_parties):
        """ Parses a line to determine weather result is for n_parties """
        all_cols = line.split(self.col_separator)
        if not all_cols:
            return False
        else:
            try:
                return str(n_parties) in all_cols[1]
            except IndexError:
                return False

    def fill_value(self, approach, protocol, n_parties, val):
        """ Fills a value in the table """
        table_list = self.row_groups

        try:
            row, col = self.row_group_labels.index(protocol), self.column_labels.index(approach)
        except ValueError:
            print(f"Could not find position: {approach}, {protocol} in {self.column_labels} or {self.row_group_labels}")
            return

        tmp_list = table_list[row].split(self.n_parties_separator)

        res_list = [self.fill_runtime_pattern(x, col, val)
                    if self.line_for_n_parties(x, n_parties)
                    else x for x in tmp_list]

        table_list[row] = self.n_parties_separator.join(res_list)

        self.table = self.multirow_separator.join(table_list)



