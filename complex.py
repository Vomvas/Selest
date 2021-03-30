"""
Helper classes for complex arithmetic that wrap around other classes like sint and Matrix.
"""
from .library import print_str, print_ln, for_range, for_range_parallel, for_range_opt
from .types import *
from .types import _int
from .types import _secret
from Compiler.GC.types import *
from .mpc_math import sqrt, abs_fx
from Compiler.exceptions import CompilerError
from Compiler import util

# v_type = sbitfix
v_type = sfix


class cplx(object):
    """
    Complex number. Defines properties and operations.
    """

    __slots__ = ['re', 'im', 'size']

    @staticmethod
    def n_elements():
        return 2

    @staticmethod
    def mem_size():
        return 1

    @classmethod
    def set_precision(cls, *args):
        """Set precision of secret scalars and their revealed form"""
        cls.value_type.set_precision(*args)

    @classmethod
    def is_address_tuple(cls, address):
        if isinstance(address, (list, tuple)):
            assert (len(address) == cls.n_elements())
            return True
        return False

    def is_secret(self):
        return isinstance(self.re, _secret) or isinstance(self.im, _secret)

    def __iter__(self):
        yield self.re
        yield self.im

    def __eq__(self, other):
        if isinstance(other, cplx):
            return (self.re==other.re)*(self.im==other.im)
        elif isinstance(other, (cfix, sfix, sint, sfloat, regint, cint, sint, sbitfix)):
            return (self.re == other)*(self.im == self.value_type(0))
        elif isinstance(other, complex):
            return (self.re == other.real)*(self.imag==other.imag)
        else:
            raise NotImplementedError

    def __ne__(self, other):
        return 1-self.__eq__(other)

    def __neg__(self):
        return self._new(-self.re, -self.im)

    def __invert__(self):
        magn_sq = self.square_abs()
        re_part = self.re / magn_sq
        im_part = -self.im / magn_sq
        return self._new(re_part, im_part)

    def __add__(self, other):
        if is_zero(other):
            return self
        else:
            if self.value_type in [sbitfix, cbitfix]:
                return self._add_bitfix(other)
            return self.add(other)

    __radd__ = __add__

    def __sub__(self, other):
        if is_zero(other):
            return self
        else:
            return self.sub(other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __len__(self):
        assert len(self.re) == len(self.im)
        return len(self.re)

    def __mul__(self, other):
        return self.mul(other)

    __rmul__ = __mul__


class ccplx(cplx):
    """
    Clear complex number. Helper class for printing revealed scplx.
    """

    if v_type == sfix:
        value_type = cfix
    elif v_type == sbitfix:
        value_type = cbitfix

    @classmethod
    def set_value_type(cls, v_type):
        assert v_type in [cint, cfix, cbitfix]
        cls.value_type = v_type

    @classmethod
    def _new(cls, other_re, other_im):
        res = cls(other_re, other_im)
        return res

    def __init__(self, real=0, imag=0, size=1):
        if (self.value_type == cint and
            util.is_constant_float(real)
            and util.is_constant_float(imag)):
            real = int(real)
            imag = int(imag)
        self.re = self.value_type.conv(real)
        self.im = self.value_type.conv(imag)
        self.size = size

    @classmethod
    def malloc(cls, size, creator_tape=None):
        return program.malloc(size * cls.n_elements(), cint, creator_tape=creator_tape)

    @classmethod
    @read_mem_value
    def conv(cls, other):
        if isinstance(other, cls):
            return other
        else:
            try:
                return(cls.load_other(other))
            except NotImplementedError as e:
                raise e

    @classmethod
    def load_other(cls, other):
        if isinstance(other, scplx) or isinstance(other, _secret):
            raise TypeError("Cannot load secret value <%s> in clear complex." % type(other))
        elif isinstance(other, ccplx):
            return ccplx(other.re, other.im)
        elif isinstance(other, (cfix, regint, cint, cbitfix)):
            return ccplx(other, 0)
        elif isinstance(other, complex):
            return ccplx(other.real, other.imag)
        else:
            raise NotImplementedError("Cannot load type %s into type ccplx." % type(other))

    @vectorized_classmethod
    def load_mem(cls, address, mem_type=None):
        """ Load from memory by public address. """
        size = get_global_vector_size()
        if cls.is_address_tuple(address):
            return ccplx(*(cls.value_type.load_mem(a, size=size) for a in address))
        res = []
        for i in range(2):
            res.append(cls.value_type.load_mem(address + i * size, size=size))
        return ccplx(*res)

    def store_in_mem(self, address):
        """ Store in memory by public address. """
        if self.is_address_tuple(address):
            for a, x in zip(address, self):
                x.store_in_mem(a)
            return
        for i,x in enumerate((self.re, self.im)):
            x.store_in_mem(address + i * self.size)

    def conj(self):
        """Returns the complex conjugate of the complex number"""
        return ccplx(self.re, -self.im)

    def _add_bitfix(self, other):
        if isinstance(other, scplx):
            return scplx(self.re + other.re, self.im + other.im)
        elif isinstance(other, ccplx):
            return ccplx(self.re + other.re, self.im + other.im)
        elif isinstance(other, (int, cint, sint)):
            return ccplx(self.re + other, self.im)
        else:
            return NotImplemented

    @vectorize
    def add(self, other):
        # other = self.coerce(other)
        if isinstance(other, scplx):
            return scplx(self.re + other.re, self.im + other.im)
        elif isinstance(other, ccplx):
            return ccplx(self.re + other.re, self.im + other.im)
        elif isinstance(other, (int, cint, sint)):
            return ccplx(self.re + other, self.im)
        else:
            return NotImplemented

    # @vectorize
    def sub(self, other):
        if isinstance(other, scplx):
            return scplx(self.re - other.re, self.im - other.im)
        elif isinstance(other, ccplx):
            return ccplx(self.re - other.re, self.im - other.im)
        elif isinstance(other, (int, cint, sint)):
            return ccplx(self.re - other, self.im)
        else:
            return NotImplemented

    def mul_3m(self, other):
        # Compute the require   d scalar multiplications once and follow the 3M method
        if isinstance(other, cplx):
            t1 = self.re * other.re
            t2 = self.im * other.im
            t3 = (self.re + self.im) * (other.re + other.im)
            if other.is_secret():
                return scplx(t1 - t2, t3 - t1 - t2)
            else:
                return ccplx(t1 - t2, t3 - t1 - t2)
        elif isinstance(other, complex):
            return self.mul_3m(ccplx.load_other(complex))
        else:
            raise NotImplementedError

    def __abs__(self):
        """ Clear complex magnitude """
        return sqrt(self.re.square() + self.im.square())

    def mul(self, other):
        """ Uses the 3M method (3 scalar multiplications) for complex multiplication"""
        if isinstance(other, (ccplx, complex)):
            return self.mul_3m(ccplx.load_other(other))
        elif isinstance(other, (int, cint, regint, cfix, float, cfloat)):
            return ccplx(other * self.re, other * self.im, size=self.size)
        elif isinstance(other, (sint, sfix, sfloat)):
            return scplx(other * self.re, other * self.im, size=self.size)
        elif isinstance(other, (scplx, CplxArray, CplxMatrix)):
            return other * self
        elif isinstance(other, Array):
            if isinstance(other.value_type, _secret):
                res_type = scplx
            else:
                res_type = ccplx
            res = CplxArray(other.length, value_type=res_type)

            if self.value_type == sbitfix:
                @for_range(res.length)
                def _(i):
                    res[i] = self * other[i]
            else:
                @for_range_parallel(res.length, res.length)
                def _(i):
                    res[i] = self * other[i]

            return res
        elif isinstance(other, Matrix):
            if isinstance(other.value_type, _secret):
                res_type = scplx
            else:
                res_type = ccplx
            res = CplxMatrix(other.sizes[0], other.sizes[1], value_type=res_type)

            if self.value_type == sbitfix:
                @for_range(res.rows)
                def _(i):
                    res.set_row(i, self * other[i])
            else:
                @for_range_parallel(res.rows, res.rows)
                def _(i):
                    res.set_row(i, self * other[i])

            return res
        else:
            raise TypeError("Multiplication of type <complex.cplx> with type of <%s> is not supported" % type(other))

    def __truediv__(self, other):
        if isinstance(other, _secret):
            inv = 1 / other
            return scplx(self.re * inv, self.im * inv)
        else:
            return ccplx(self.re / other, self.im / other)

    def output(self):
        """ Clear complex number output. """
        if self.value_type == cbitfix:
            print_char("(")
            v = self.re.v
            if self.re.k < v.unit:
                bits = self.re.v.bit_decompose(self.re.k)
                sign = bits[-1]
                v += (sign << (self.re.k)) * -1
            inst.print_float_plainb(v, cbits(-self.re.f, n=32), cbits(0), cbits(0), cbits(0))
            print_str(", ")
            v = self.im.v
            if self.im.k < v.unit:
                bits = self.im.v.bit_decompose(self.im.k)
                sign = bits[-1]
                v += (sign << (self.im.k)) * -1
            inst.print_float_plainb(v, cbits(-self.im.f, n=32), cbits(0), cbits(0), cbits(0))
            print_str("j)")
        elif self.value_type == cfix:
            print_char("(")
            tmp_re = regint()
            convmodp(tmp_re, self.re.v, bitlength=self.re.k)
            sign_re = cint(tmp_re < 0)
            abs_v_re = sign_re.if_else(-self.re.v, self.re.v)
            print_float_plain(cint(abs_v_re), cint(-self.re.f), \
                              cint(0), cint(sign_re), cint(0))
            tmp_im = regint()
            convmodp(tmp_im, self.im.v, bitlength=self.im.k)
            sign_im = cint(tmp_im < 0)
            abs_v_im = sign_im.if_else(-self.im.v, self.im.v)

            (1 - sign_im).print_if(" + ")

            print_float_plain(cint(abs_v_im), cint(-self.im.f), \
                              cint(0), cint(sign_im), cint(0))
            print_char("j")
            print_char(")")
        elif self.value_type == cint:
            print_char('(')
            tmp_re = regint()
            tmp_re.load_other(self.re)
            print_int(tmp_re)
            print_str(", ")
            tmp_im = regint()
            tmp_im.load_other(self.im)
            print_int(tmp_im)
            print_char(')')


    def print_reveal(self, no_new_line=False):
        """Unpack complex number and print it"""
        library.break_point("print_cplx")
        print_str("%s", self)
        if not no_new_line:
            print_str("\n")
        library.break_point("print_cplx")


class scplx(cplx):
    """
        Secret complex number.
        Represents (re + im * j).

            re: real part

            im: imaginary part

        This uses secret fixed arithmetic internally.
    """
    __slots__ = ['re', 'im', 'size']

    clear_type = ccplx
    value_type = v_type
    value_clear_type = v_type.clear_type

    @classmethod
    def set_value_type(cls, v_type):
        """Changes the value type of the class for the real and imaginary parts (default is sfix)"""
        assert v_type in [sint, sfix, sbitfix]
        cls.value_type = v_type
        ccplx.set_value_type(v_type.clear_type)

    @classmethod
    def _new(cls, other_re, other_im):
        res = cls(other_re, other_im)
        return res

    @classmethod
    def j(cls):
        """Returns the imaginary unit"""
        return cls(0, 1)

    @staticmethod
    def are_conjugate(a, b):
        """Returns true if a and b are conjugates"""
        res = (a.re==b.re)*(a.im==-b.im)
        return res

    @classmethod
    def load_other(cls, other):
        if isinstance(other, scplx):
            return scplx(other.re, other.im)
        elif isinstance(other, (int, cint, regint, sint, sfix, cfix, sbitfix)):
            return scplx(other, 0)
        elif isinstance(other, complex):
            return scplx(other.real, other.imag)
        else:
            raise NotImplementedError("Cannot load type %s into type scplx." % type(other))

    @classmethod
    def get_input_from(cls, player_id):
        """Loads player input"""
        real_part = cls.value_type.get_input_from(player_id)
        imag_part = cls.value_type.get_input_from(player_id)
        return cls(real_part, imag_part)

    @classmethod
    def malloc(cls, size, creator_tape=None):
        try:
            return program.malloc(size * cls.n_elements(), cls.value_type.int_type, creator_tape=creator_tape)
        except AttributeError:
            return program.malloc(size * cls.n_elements(), sint, creator_tape=creator_tape)
        except AttributeError:
            raise NotImplementedError

    @classmethod
    @read_mem_value
    def conv(cls, other):
        if isinstance(other, cls):
            return other
        else:
            try:
                return(cls.load_other(other))
            except NotImplementedError as e:
                raise e

    @classmethod
    def coerce(cls, other):
        if isinstance(other, scplx):
            return other
        else:
            return cls.conv(other)

    @vectorized_classmethod
    def load_mem(cls, address, mem_type=None):
        """ Load from memory by public address. """
        size = get_global_vector_size()
        if cls.is_address_tuple(address):
            return scplx(*(cls.value_type.load_mem(a, size=size) for a in address), size=size)
        res = []
        for i in range(2):
            res.append(cls.value_type.load_mem(address + i * size, size=size))
        try:
            return scplx(*res)
        except CompilerError as e:
            print("Size in load mem: ", size, [x.size for x in res])
            raise e

    def store_in_mem(self, address):
        """ Store in memory by public address. """
        if self.is_address_tuple(address):
            for a, x in zip(address, self):
                x.store_in_mem(a)
            return
        for i,x in enumerate((self.re, self.im)):
            x.store_in_mem(address + i * self.size)

    def __init__(self, real=0, imag=0, size=None):
        # self.size = get_global_vector_size()
        try:
            if size is not None:
                self.size = size
            else:
                self.size = real.size
        except AttributeError:
            self.size = 1
        try:
            if self.value_type == sbitfix and self.size not in [None, 1]:
                raise NotImplementedError
            else:
                if (self.value_type == sint and
                    util.is_constant_float(real)
                    and util.is_constant_float(imag)):
                    real = int(real)
                    imag = int(imag)
                self.re = self.value_type(real)
                self.im = self.value_type(imag)
        except CompilerError as e:
            raise CompilerError("Invalid types for real and imaginary parts: %s, %s\nError: %s" % (real, imag, e))

    def _add_bitfix(self, other):
        if isinstance(other, (scplx, ccplx)):
            return scplx(self.re + other.re, self.im + other.im)
        elif isinstance(other, (int, cint, sint)):
            return scplx(self.re + other, self.im)
        else:
            return NotImplemented

    @vectorize
    def add(self, other):
        # other = self.coerce(other)
        if isinstance(other, (scplx, ccplx)):
            return scplx(self.re + other.re, self.im + other.im)
        elif isinstance(other, (int, cint, sint)):
            return scplx(self.re + other, self.im)
        else:
            return NotImplemented

    @vectorize
    def sub(self, other):
        if isinstance(other, scplx):
            return scplx(self.re - other.re, self.im - other.im)
        elif isinstance(other, (int, cint, sint)):
            return scplx(self.re - other, self.im)
        else:
            return NotImplemented

    def mul_3m(self, other):
        if isinstance(other, cplx):
            # Compute the required scalar multiplications once and follow the 3M method
            t1 = self.re * other.re
            t2 = self.im * other.im
            t3 = (self.re + self.im) * (other.re + other.im)
        elif isinstance(other, complex):
            return self.mul_3m(ccplx.load_other(complex))
        else:
            return NotImplemented
        return scplx(t1 - t2, t3 - t1 - t2)

    def mul(self, other):
        """ Uses the 3M method (3 scalar multiplications) for complex multiplication"""
        if isinstance(other, (cplx, complex)):
            return self.mul_3m(other)
        elif isinstance(other, (int, cint, regint, sint, cfix, sfix, float, cfloat, sfloat, sbitfix, sbit)):
            return scplx(other * self.re, other * self.im, size=self.size)
        elif isinstance(other, (CplxArray, CplxMatrix)):
            return other * self
        elif isinstance(other, Array):
            res = CplxArray(other.length, value_type=scplx)

            if self.value_type == sbitfix:
                @for_range(res.length)
                def _(i):
                    res[i] = self * other[i]
            else:
                @for_range_parallel(res.length, res.length)
                def _(i):
                    res[i] = self * other[i]

            return res
        elif isinstance(other, Matrix):
            res = CplxMatrix(other.sizes[0], other.sizes[1], value_type=scplx)

            if self.value_type == sbitfix:
                @for_range(res.rows)
                def _(i):
                    res.set_row(i, self * other[i])
            else:
                @for_range_parallel(res.rows, res.rows)
                def _(i):
                    res.set_row(i, self * other[i])

            return res
        else:
            raise TypeError("Multiplication of type <Complex> with type of <%s> is not supported" % type(other))

    def __truediv__(self, other):
        if isinstance(other, (int, cint, regint, sint, cfix, sfix, float, cfloat, sfloat)):
            inv = 1 / other
            return scplx(self.re * inv, self.im * inv)
        else:
            raise NotImplementedError

    def inverse(self):
        """Returns the inverse of the complex number"""
        return NotImplemented

    def conj(self):
        """Returns the complex conjugate of the complex number"""
        return scplx(self.re, -self.im)

    def conjugate(self):
        """Returns the complex conjugate of the complex number"""
        return self.conj()

    def __abs__(self):
        """Returns the absolute of the complex number"""
        return sqrt(self.re.square() + self.im.square())

    def abs_l1(self):
        """ Returns the l1 norm of the complex number """
        return abs_fx(self.re) + abs_fx(self.im)

    def square_abs(self):
        """
        Returns the squared absolute of the complex number. Avoids the complexity of square rooting; equivalent to
        multiplying with the conjugate.
        """
        return self.re.square() + self.im.square()

    def print_reveal(self, no_new_line=False):
        """Unpack complex number and print it"""
        library.break_point("print_cplx")
        print_str("%s", self.reveal())
        if not no_new_line:
            print_str("\n")
        library.break_point("print_cplx")

    def reveal(self):
        """ Reveal secret complex number.

        :return: ccplx """
        return ccplx(self.re.reveal(), self.im.reveal())


class CplxArray(Array):
    """
    Based on Compiler.Array
    """

    @classmethod
    def zeros(cls, length, v_type=scplx):
        res = cls(length, v_type)
        res.assign_all(res.value_type.value_type(0))
        return res

    def __init__(self, length, value_type=scplx, address=None, debug=None, alloc=True, real_arr=None, imag_arr=None):

        assert value_type in [scplx, ccplx]

        self.value_type = value_type

        super(CplxArray, self).__init__(length, value_type, address=address, debug=debug, alloc=alloc)

        if real_arr is not None and imag_arr is not None:
            if isinstance(real_arr, Array) and isinstance(imag_arr, Array):
                assert real_arr.length == imag_arr.length == self.length
                assert real_arr.value_type == imag_arr.value_type == v_type
            else:
                raise TypeError("Real and imaginary arrays must be Array type of %s." % v_type)

            @for_range_opt(self.length)
            def _(i):
                self[i] = scplx(real_arr[i], imag_arr[i])

    @property
    def reals(self):
        return self._get_reals_array()

    @property
    def imags(self):
        return self._get_imags_array()

    @classmethod
    def create_from(cls, l):
        if isinstance(l, cls):
            return l
        tmp = list(l)
        res = CplxArray(len(tmp), value_type=type(tmp[0]))
        res.assign(tmp)
        return res

    def assign_all(self, value, use_threads=True, conv=True):
        if self.value_type.value_type == sbitfix:
            @for_range(self.length)
            def _(i):
                self[i] = self.value_type.load_other(value)
        else:
            super(CplxArray, self).assign_all(value, use_threads=use_threads, conv=conv)

    def assign(self, other):
        if self.value_type.value_type == sbitfix:
            assert self.length == len(other)
            if isinstance(other, CplxArray):
                @library.for_range(len(other))
                def _(i):
                    self[i] = other[i]
            else:
                for i, j in enumerate(other):
                    self[i] = j
        else:
            super(CplxArray, self).assign(other)

    def _get_reals_array(self):
        """
        Returns an array of scalars that correspond to the real parts. The value type of these scalars
        is not complex.
        """
        return Array.create_from([x.re for x in self])

    def _get_imags_array(self):
        """
        Returns an array of scalars that correspond to the imaginary parts. The value type of these scalars
        is not complex.
        """
        return Array.create_from([x.im for x in self])

    def delete(self):
        self.reals.delete()
        self.imags.delete()

    def __add__(self, other):
        if self.value_type.value_type == sbitfix:
            return CplxArray.create_from([x + y for (x, y) in zip(self, other)])
        else:
            return super(CplxArray, self).__add__(other)

    def __sub__(self, other):
        """ Complex vector subtraction.

        :param other: vector or container of same length and type that supports operations with type of this array """
        assert len(self) == len(other)
        if self.value_type.value_type == sbitfix:
            return CplxArray.create_from([x - y for (x, y) in zip(self, other)])
        else:
            return self.get_vector() - other

    def __neg__(self):
        if is_zero(self):
            return self
        res = CplxArray.create_from([-x for x in self])
        return res

    def arr_mult(self, other):
        """ Array multiplication (different than complex vector dot product) """
        assert self.length == other.length
        if isinstance(self.value_type, ccplx) and isinstance(other.value_type, ccplx):
            res_type = ccplx
        else:
            res_type = scplx
        products = CplxArray(self.length, value_type=res_type)
        if self.value_type.value_type == sbitfix:
            @for_range(self.length)
            def _(i):
                products[i] = self[i] * other[i]
        else:
            @for_range_parallel(self.length, self.length)
            def _(i):
                products[i] = self[i] * other[i]
        return sum(products)

    def arr_mat_mult(self, other):
        """ Vector matrix multiplication """
        if isinstance(self.value_type, ccplx) and isinstance(other.value_type, ccplx):
            res_type = ccplx
        else:
            res_type = scplx
        res = CplxArray(other.sizes[1], value_type=res_type)
        if self.value_type.value_type == sbitfix:
            @for_range(other.sizes[1])
            def _(j):
                res[j] = self.arr_mult(other.get_column(j))
        else:
            @for_range_parallel(other.sizes[1], other.sizes[1])
            def _(j):
                res[j] = self.arr_mult(other.get_column(j))
        return res

    def __mul__(self, other):
        if isinstance(other, CplxArray):
            assert self.length == other.length
            return self.arr_mult(other)
        elif isinstance(other, CplxMatrix):
            assert self.length == other.sizes[0]
            return self.arr_mat_mult(other)
        elif isinstance(other, (int, cint, regint, sint, cfix, sfix, float, cfloat, sfloat)):
            return self.get_vector() * other
        elif isinstance(other, (sbitfix, sbit)):
            # Sbitfix is not intended for large arithmetic circuits so Python loop unrolling should be acceptable.
            return CplxArray.create_from([x * other for x in self])
        elif isinstance(other, cplx):
            res = CplxArray(self.length)
            if other.value_type == sbitfix:
                @for_range(res.length)
                def _(i):
                    res[i] = self[i] * other
                return res
            else:
                if self.value_type.value_type == sbitfix:
                    @for_range(res.length)
                    def _(i):
                        res[i] = self[i] * other
                else:
                    @for_range_parallel(res.length, res.length)
                    def _(i):
                        res[i] = self[i] * other
                return res
        else:
            raise NotImplementedError("Multiply operation not implemented for types %s and %s" % (type(self),
                                                                                                  type(other)))
    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, regint, cint, sint, cfix, sfix, sfloat, sbitfix)):
            return self.__mul__(1 / other)
        else:
            raise NotImplementedError

    # __radd__ = __add__

    def __rsub__(self, other):
        return -self.__sub__(other)

    def conjugate(self):
        """ Element-wise conjugate """
        return CplxArray.create_from([x.conj() for x in self])

    def norm_squared(self):
        """Avoids complexity of square root, equivalent to self dot product"""
        absolutes = Array(self.length, self.value_type.value_type)

        if self.value_type.value_type == sbitfix:
            @for_range(absolutes.length)
            def _(i):
                absolutes[i] = self[i].square_abs()
        else:
            @for_range_parallel(absolutes.length, absolutes.length)
            def _(i):
                # absolutes[regint(i)] = self[regint(i)].square_abs()
                absolutes[i] = self[i].square_abs()
        norm_squared = sum(absolutes)
        return norm_squared
    
    def norm(self, type="l2"):
        """Euclidean norm"""
        if type=="l2":
            return sqrt(self.norm_squared())
        elif type=="l1":
            dists = Array(self.length, self.value_type.value_type)

            if self.value_type.value_type == sbitfix:
                @for_range(dists.length)
                def _(i):
                    # dists[i] = abs(self[i])
                    dists[i] = (self[i]).abs_opt()
            else:
                @for_range_parallel(dists.length, dists.length)
                def _(i):
                    # dists[i] = abs(self[i])
                    dists[i] = (self[i]).abs_opt()

            l1_norm = sum(dists)
            return l1_norm
        elif type=="squared":
            return self.norm_squared()
        else:
            raise NotImplementedError

    def _covariance_sbitfix(self, other=None):
        """
        Handles covariance calculation in the case of sbitfix
        """
        if other is not None:
            raise NotImplementedError
        res = CplxMatrix(self.length, self.length)
        @for_range(res.rows)
        def _(i):
            res[i][i] = self[i].square_abs()
            @for_range(i)
            def _(j):
                res[i][j] = self[i] * self[j].conjugate()
                res[j][i] = res[i][j].conjugate()
        return res

    def covariance(self, other=None):
        """
        Returns the covariance of two complex vectors. If no other vector is provided, returns the covariance with
        itself.
        """
        if self.value_type.value_type == sbitfix:
            return self._covariance_sbitfix()
        if other is None:
            res = CplxMatrix(self.length, self.length)
            @for_range_parallel(res.rows, res.rows)
            def _(i):
                res[i][i] = self[i].square_abs()
                @for_range_parallel(i, i)
                def _(j):
                    res[i][j] = self[i] * self[j].conjugate()
                    res[j][i] = res[i][j].conjugate()
        else:
            assert self.length == other.length
            res = CplxMatrix(self.length, self.length)
            @for_range_parallel(res.rows, res.rows)
            def _(i):
                res[i][i] = self[i] * other[i].conjugate()
                @for_range_parallel(i, i)
                def _(j):
                    res[i][j] = self[i] * other[j].conjugate()
                    res[j][i] = res[i][j].conjugate()
        return res

    def print_reveal(self):
        """Unpack complex number and print it"""
        library.break_point("print_cplx_arr")
        print_str("[")
        @for_range_opt(self.length)
        def _(i):
            self[i].print_reveal(no_new_line=True)
            (i < self.length - 1).print_if(", ")
        print_ln("]")
        library.break_point("print_cplx_arr")

    def reveal(self):
        """ Reveal the whole array.

        :returns: Array of relevant clear type. """
        return CplxArray.create_from(x.reveal() for x in self)


class CplxSubMultiArray(SubMultiArray):
    """ Multidimensional array functionality for complex numbers. """

    def __init__(self, sizes, address, index, value_type=scplx, debug=None):

        assert value_type in [scplx, ccplx]

        self.value_type = value_type

        super(CplxSubMultiArray, self).__init__(sizes, self.value_type, address, index, debug)

    def __getitem__(self, index):
        """ Part access.

        :param index: public (regint/cint/int)
        :return: :py:class:`Array` if one-dimensional, :py:class:`SubMultiArray` otherwise"""
        if util.is_constant(index) and index >= self.sizes[0]:
            raise StopIteration
        key = program.curr_block, str(index)
        if key not in self.sub_cache:
            if self.debug:
                library.print_ln_if(index >= self.sizes[0], \
                                    'OF%d:' % len(self.sizes) + self.debug)
            if len(self.sizes) == 2:
                self.sub_cache[key] = \
                        CplxArray(self.sizes[1], self.value_type, \
                              self.address + index * self.sizes[1] *
                              self.value_type.n_elements(), \
                              debug=self.debug)
            else:
                self.sub_cache[key] = \
                        CplxSubMultiArray(self.sizes[1:], \
                                      self.address, index, value_type=self.value_type, debug=self.debug)
        return self.sub_cache[key]

    def get_vector(self, base=0, size=None):
        """ Return vector with content. Not implemented for floating-point.

        :param base: public (regint/cint/int)
        :param size: compile-time (int) """
        assert self.value_type.n_elements() == 2
        size = size or self.total_size()
        return self.value_type.load_mem(self.address + base, size=size)

    def assign_all(self, value):
        if self.value_type.value_type == sbitfix:
            @for_range(self.sizes[0])
            def _(i):
                @for_range(self.sizes[1])
                def _(j):
                    self[i][j] = self.value_type.load_other(value)
        else:
            super(CplxSubMultiArray, self).assign_all(value)

    def assign_vector(self, vector, base=0):
        """ Assign vector to content.

        :param vector: vector of matching size convertible to relevant basic type
        :param base: compile-time (int) """
        assert self.value_type.n_elements() == 2
        assert vector.size <= self.total_size()
        vector.store_in_mem(self.address + base)

    def same_shape(self):
        """ :return: new multidimensional array with same shape and basic type """
        return CplxMultiArray(self.sizes, self.value_type)

    def input_from(self, player, budget=None, raw=False):
        """ Fill with inputs from player if supported by type.

        :param player: public (regint/cint/int) """
        budget = budget or Tape.Register.maximum_size
        if (self.total_size() < budget) and \
           self.value_type.n_elements() == 1:
            if raw or program.always_raw():
                input_from = self.value_type.get_raw_input_from
            else:
                input_from = self.value_type.get_input_from
            self.assign_vector(input_from(player, size=self.total_size()))
        else:
            @library.for_range_opt(self.sizes[0],
                                   budget=budget / self[0].total_size())
            def _(i):
                self[i].input_from(player, budget=budget, raw=raw)

    def schur(self, other):
        """ Element-wise product.

        :param other: container of matching size and type
        :return: container of same shape and type as :py:obj:`self` """
        assert self.sizes == other.sizes
        if len(self.sizes) == 2:
            res = CplxMatrix(self.sizes[0], self.sizes[1], self.value_type)
        else:
            res = CplxMultiArray(self.sizes, self.value_type)
        res.assign_vector(self.get_vector() * other.get_vector())
        return res

    def __neg__(self):
        if is_zero(self):
            return self
        if len(self.sizes) == 2:
            res = CplxMatrix(self.sizes[0], self.sizes[1], self.value_type)
        else:
            res = CplxMultiArray(self.sizes, self.value_type)
        res.assign_vector(-self.get_vector())
        return res

    def __add__(self, other):
        """ Element-wise addition.

        :param other: container of matching size and type
        :return: container of same shape and type as :py:obj:`self` """
        if is_zero(other):
            return self
        assert self.sizes == other.sizes
        if len(self.sizes) == 2:
            res = CplxMatrix(self.sizes[0], self.sizes[1], self.value_type)
        else:
            res = CplxMultiArray(self.sizes, self.value_type)
        res.assign_vector(self.get_vector() + other.get_vector())
        return res

    __radd__ = __add__

    def __sub__(self, other):
        """ Element-wise subtraction.

                :param other: container of matching size and type
                :return: container of same shape and type as :py:obj:`self` """
        if is_zero(other):
            return self
        assert self.sizes == other.sizes
        if len(self.sizes) == 2:
            res = CplxMatrix(self.sizes[0], self.sizes[1], self.value_type)
        else:
            res = CplxMultiArray(self.sizes, self.value_type)
        res.assign_vector(self.get_vector() - other.get_vector())
        return res

    def __rsub__(self, other):
        return - self + other

    def iadd(self, other):
        """ Element-wise addition in place.

        :param other: container of matching size and type """
        assert self.sizes == other.sizes
        self.assign_vector(self.get_vector() + other.get_vector())

    def __mul__(self, other):
        """ Matrix-matrix and matrix-vector multiplication.

        :param self: two-dimensional
        :param other: Matrix or Array of matching size and type """
        return self.mul(other)

    def transpose(self):
        """ Matrix transpose.

        :param self: two-dimensional """
        assert len(self.sizes) == 2
        res = CplxMatrix(self.sizes[1], self.sizes[0], self.value_type)
        library.break_point()
        @library.for_range_opt(self.sizes[1])
        def _(i):
            @library.for_range_opt(self.sizes[0])
            def _(j):
                res[i][j] = self[j][i]
        library.break_point()
        return res

    def reveal_list(self):
        """ Reveal as list. """
        return list(self.get_vector().reveal())

    def reveal_nested(self):
        """ Reveal as nested list. """
        flat = iter(self.get_vector().reveal())
        res = []
        def f(sizes):
            if len(sizes) == 1:
                return [next(flat) for i in range(sizes[0])]
            else:
                return [f(sizes[1:]) for i in range(sizes[0])]
        return f(self.sizes)


class CplxMultiArray(CplxSubMultiArray):
    """ Multidimensional array. """

    def __init__(self, sizes, value_type=scplx, debug=None, address=None, alloc=True):
        """
        :param sizes: shape (compile-time list of integers)
        :param value_type: basic type of entries
        """
        assert value_type in [scplx, ccplx]

        self.value_type = value_type

        if isinstance(address, CplxArray):
            self.array = address
        else:
            self.array = CplxArray(reduce(operator.mul, sizes), value_type=self.value_type, address=address, alloc=alloc)
        CplxSubMultiArray.__init__(self, sizes, self.array.address, 0, value_type=self.value_type, \
                               debug=debug)
        if len(sizes) < 2:
            raise CompilerError('Use CplxArray')

    @property
    def address(self):
        return self.array.address

    @address.setter
    def address(self, value):
        self.array.address = value

    def alloc(self):
        self.array.alloc()

    def delete(self):
        self.array.delete()


class CplxMatrix(CplxMultiArray):
    """
    A wrapper around Matrix class that enables storing complex numbers by using two matrices, one for each complex
    element.
    :param rows:
    :param col:
    """

    @classmethod
    def unit_mat(cls, length):
        """ Returns a complex vector v with norm 1 (simulates preprocessed data) """
        from scipy.stats import unitary_group
        unit = unitary_group.rvs(length)
        res = cls(length, length)
        for i in range(unit.shape[0]):
            for j in range(unit.shape[1]):
                res[i][j] = unit[i][j]
        return res

    @classmethod
    def zeros(cls, nrows, ncols, value_type=scplx):
        """Returns an all-zero complex matrix"""
        res = cls(nrows, ncols, value_type=value_type)
        res.assign_all(0)
        return res

    @classmethod
    def eye(cls, n, value_type=scplx):
        """Returns an n x n Identity matrix"""
        res = cls.zeros(n, n, value_type=value_type)

        @for_range_opt(res.rows)
        def _(i):
            res[i][i] = res.value_type.value_type(1)
        return res

    def __init__(self, rows, col, value_type=scplx, real_mat=None, imag_mat=None, address=None):

        assert value_type in [scplx, ccplx]

        self.value_type = value_type

        super(CplxMatrix, self).__init__([rows, col], value_type=self.value_type, address=address)
        self.rows = rows
        self.col = col

        if real_mat is not None and imag_mat is not None:
            @for_range_opt(self.rows)
            def _(i):
                @for_range_opt(self.col)
                def _(j):
                    self[i][j] = scplx(real_mat[i][j], imag_mat[i][j])

    @property
    def reals(self):
        return self._get_reals_matrix()

    @property
    def imags(self):
        return self._get_imags_matrix()

    def _get_reals_matrix(self):
        res = Matrix(self.rows, self.col, self.value_type.value_type)
        @for_range_opt(self.sizes[0])
        def _(i):
            # res[i] = Array.create_from([x.re for x in self[i]])
            res[i].assign_vector(Array.create_from([x.re for x in self[i]]).get_vector())
            # tmp = Array.create_from([x.re for x in self[i]])
            # res[i] = tmp
        return res

    def _get_imags_matrix(self):
        res = Matrix(self.rows, self.col, self.value_type.value_type)
        @for_range_opt(self.sizes[0])
        def _(i):
            res[i].assign_vector(Array.create_from([x.im for x in self[i]]).get_vector())
        return res

    def get_row(self, index):
        """Helper function for backwards compatibility; equivalent to matrix indexing: M[i] """
        return self[index]

    def set_row(self, index, other):
        """
        Set a matrix row
        :param index:
        :param other: complex.CplxArray
        :return:
        """
        if isinstance(other, CplxArray):
            assert len(other) == self.sizes[1]
            self[index].assign(other)
        elif isinstance(other, scplx):
            assert other.size == self.sizes[1]
            self[index].assign(other)
        else:
            raise NotImplementedError

    def get_column(self, index):
        res = CplxArray(self.sizes[0], value_type=self.value_type)
        if self.value_type.value_type == sbitfix:
            @library.for_range(res.length)
            def _(i):
                res[i] = self[i][index]
        else:
            @library.for_range_opt(res.length)
            def _(i):
                res[i] = self[i][index]
        return res

    def set_column(self, index, other):
        """
        Set a matrix column
        :param index:
        :param other:
        :return:
        """
        if isinstance(other, CplxArray):
            assert len(other) == self.sizes[0]
            @for_range_opt(len(other))
            def _(i):
                self[i][index] = self.value_type.load_other(other[i])
        elif isinstance(other, self.value_type):
            try:
                tmp = CplxArray(other.size, value_type=type(other))
                tmp.assign(other)
                self.set_column(index, tmp)
            except CompilerError as e:
                raise e
        elif isinstance(other, list):
            assert len(other) == self.sizes[0]
            for i in range(len(other)):
                self[i][index] = self.value_type.load_other(other[i])
        else:
            try:
                assert len(other) == self.sizes[0]
                for i in range(len(other)):
                    self[i][index] = self.value_type.load_other(other[i])
            except Exception as e:
                raise NotImplementedError(e)

    def assign(self, other):
        """Assign all elements equal to elements of given matrix (if not convertible to complex an error is raised)"""

        assert self.sizes == other.sizes

        if self.value_type.value_type == sbitfix:
            @for_range(self.sizes[0])
            def _(i):
                @for_range(self.sizes[1])
                def _(j):
                    self[i][j] = other[i][j]
            return

        if isinstance(other, CplxMatrix):
            @for_range(self.sizes[0])
            def _(i):
                self.set_row(i, other.get_row(i))
        elif isinstance(other, Matrix):
            @for_range(self.sizes[0])
            def _(i):
                @for_range(self.sizes[1])
                def _(j):
                    try:
                        self[i][j] = scplx.load_other(other[i][j])
                    except (NotImplementedError, CompilerError) as e:
                        raise e
        else:
            raise NotImplementedError

    def conjugate(self):
        """ Returns a matrix containing the complex conjugate of each original matrix element """
        res = CplxMatrix(self.rows, self.col)

        @for_range_opt(self.rows)
        def _(i):
            @for_range_opt(self.col)
            def _(j):
                res[i][j] = self[i][j].conj()
        return res

    def conj_transp(self):
        """ Returns the conjugate transpose of the matrix """
        return self.conjugate().transpose()

    def get_vector(self, base=0, size=None):
        """ Return vector with content.

        :param base: public (regint/cint/int)
        :param size: compile-time (int) """
        assert self.value_type.n_elements() == 2
        size = size or self.total_size()
        return self.value_type.load_mem(self.address + base, size=size)

    # def assign_vector(self, vector, base=0):
    #     """ Assign vector to content.
    #
    #     :param vector: vector of matching size convertible to relevant basic type
    #     :param base: compile-time (int) """
    #     assert self.value_type.n_elements() == 2
    #     assert vector.size <= self.total_size()
    #     vector.store_in_mem(self.address + base)

    def __add__(self, other):
        """Element-wise addition"""
        if is_zero(other):
            return self
        assert self.sizes == other.sizes
        res = CplxMatrix(self.sizes[0], self.sizes[1])
        if self.value_type.value_type == sbitfix:
            @for_range(res.sizes[0])
            def _(row):
                @for_range(res.sizes[1])
                def _(col):
                    res[row][col] = self[row][col] + other[row][col]
        else:
            res.assign_vector(self.get_vector() + other.get_vector())
        return res

    def scalar_mult(self, other):
        """ Scalar (element-wise) multiplication """
        res_matrix = CplxMatrix(self.sizes[0], self.sizes[1], self.value_type)
        try:
            @library.for_range_opt(res_matrix.sizes[0])
            def _(i):
                res_matrix[i].assign(self[i] * other)
            return res_matrix
        except CompilerError:
            raise NotImplementedError

    def mat_mul_3m(self, other):
        """ Uses the 3M method (3 scalar multiplications) for complex multiplication"""

        # Compute the required matrix multiplications once
        T1 = self.reals * other.reals
        T2 = self.imags * other.imags
        T3 = (self.reals + self.imags) * (other.reals + other.imags)

        # Result scalar matrices
        C1 = T1 - T2
        C2 = T3 - T1 - T2

        return CplxMatrix(self.rows, other.col, real_mat=C1, imag_mat=C2)

    def _mat_mul_sbitfix(self, other):
        res = CplxMatrix(self.rows, other.col)
        @for_range(res.rows)
        def _(row):
            @for_range(res.col)
            def _(col):
                res[row][col] = sum([x * y for (x, y) in zip(self.get_row(row), other.get_column(col))])
        return res

    def _mul_sbitfix(self, other):
        if isinstance(other, (int, float, regint, cint, sint, cfix, sfix, sfloat)):
            res = CplxMatrix(self.sizes[0], self.sizes[1])
            @for_range(res.sizes[0])
            def _(i):
                @for_range(res.sizes[1])
                def _(j):
                    res[i][j] = self[i][j] * other
            return res
        elif isinstance(other, CplxMatrix):
            return self._mat_mul_sbitfix(other)
        else:
            raise NotImplementedError

    def mul(self, other):
        if self.value_type.value_type == sbitfix:
            return self._mul_sbitfix(other)
        if isinstance(other, CplxMatrix):
            return self.mat_mul_3m(other)
        elif isinstance(other, CplxArray):
            assert self.sizes[1] == other.length
            res = CplxArray(self.sizes[0])
            @for_range_parallel(self.sizes[0], self.sizes[0])
            def _(i):
                res[i] = self.get_row(i).arr_mult(other)
            return res
        elif isinstance(other, (int, float, regint, cint, sint, cfix, sfix, sfloat)):
            return self.scalar_mult(self.value_type.value_type(other))
        elif isinstance(other, scplx):
            res = CplxMatrix(self.sizes[0], self.sizes[1])
            @for_range_parallel(res.sizes[0], res.sizes[0])
            def _(i):
                res.set_row(i, self.get_row(i) * other)
            return res
        else:
            raise NotImplementedError("Multiply operation not implemented for types %s and %s" % (type(self),
                                                                                                  type(other)))

    def scalar_div(self, other):
        return self.__mul__(1 / other)

    def __truediv__(self, other):
        if isinstance(other, (int, float, regint, cint, sint, cfix, sfix, sfloat)):
            return self.scalar_div(other)
        else:
            raise NotImplementedError

    def print_sbitfix(self):
        if self.value_type == scplx:
            print_ln('%s', [x.reveal() for y in self for x in y])
        elif self.value_type == ccplx:
            print_ln('%s', [x for y in self for x in y])

    def print_reveal(self, no_new_line=False):
        """Reveal complex matrix contents and print them"""
        library.break_point("print_cplx_mat")

        print_str("[")
        @for_range(self.rows)
        def _(i):
            print_str("[")
            @for_range(self.col)
            def _(j):
                self[i][j].print_reveal(no_new_line=True)
                library.print_str_if((j+1 < self.col), ", ")
            print_str("]")
            library.print_str_if((i + 1 < self.rows), ",\n")
        print_str("]")
        if not no_new_line:
            print_str("\n")
        library.break_point("print_cplx_mat")
