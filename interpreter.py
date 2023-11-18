import copy
from enum import Enum

from brewparse import parse_program
from env_v2 import EnvironmentManager
from intbase import InterpreterBase, ErrorType
from type_valuev2 import Type, Value, create_value, get_printable


class ExecStatus(Enum):
    CONTINUE = 1
    RETURN = 2


# Main interpreter class
class Interpreter(InterpreterBase):
    # constants
    NIL_VALUE = create_value(InterpreterBase.NIL_DEF)
    TRUE_VALUE = create_value(InterpreterBase.TRUE_DEF)
    BIN_OPS = {"+", "-", "*", "/", "==", "!=", ">", ">=", "<", "<=", "||", "&&"}

    # methods
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)
        self.trace_output = trace_output
        self.__setup_ops()

    # run a program that's provided in a string
    # usese the provided Parser found in brewparse.py to parse the program
    # into an abstract syntax tree (ast)
    def run(self, program):
        ast = parse_program(program)
        print(ast)
        print()
        self.__set_up_function_table(ast)
        self.env = EnvironmentManager()
        self.lambda_name_to_ast = {}
        self.lambdas_to_env = {}
        main_func = self.__get_func_by_name("main", 0, False)
        self.__run_statements(main_func.get("statements"))

    def __set_up_function_table(self, ast):
        self.func_name_to_ast = {}
        for func_def in ast.get("functions"):
            func_name = func_def.get("name")
            num_params = len(func_def.get("args"))
            if func_name not in self.func_name_to_ast:
                self.func_name_to_ast[func_name] = {}
            self.func_name_to_ast[func_name][num_params] = func_def

    def __get_func_by_name(self, name, num_params, called_by_var):
        if name not in self.func_name_to_ast:
            super().error(ErrorType.NAME_ERROR, f"Function {name} not found")
        candidate_funcs = self.func_name_to_ast[name]
        if num_params not in candidate_funcs:
            if (called_by_var):
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Function {name} taking {num_params} params not found",
                )
            super().error(
                ErrorType.NAME_ERROR,
                f"Function {name} taking {num_params} params not found",
            )
        return candidate_funcs[num_params]
    
    def __get_func_by_name_for_var(self, name):
        if name not in self.func_name_to_ast:
            super().error(ErrorType.NAME_ERROR, f"Function {name} not found")
        candidate_funcs = self.func_name_to_ast[name]
        if len(candidate_funcs) > 1:
            super().error(ErrorType.NAME_ERROR, f"Ambiguous function asssignment to variable")
        # print(f"candidate funcs: {candidate_funcs}")

        return candidate_funcs[list(candidate_funcs.keys())[0]]

    def __run_statements(self, statements):
        self.env.push()
        for statement in statements:
            print("----\nstatement: " + str(statement))
            if self.trace_output:
                print(statement)
            status = ExecStatus.CONTINUE
            if statement.elem_type == InterpreterBase.FCALL_DEF:
                self.__call_func(statement)
            elif statement.elem_type == "=":
                self.__assign(statement)
            elif statement.elem_type == InterpreterBase.RETURN_DEF:
                status, return_val = self.__do_return(statement)
            elif statement.elem_type == Interpreter.IF_DEF:
                status, return_val = self.__do_if(statement)
            elif statement.elem_type == Interpreter.WHILE_DEF:
                status, return_val = self.__do_while(statement)

            print("\ndictionary after running statement:")
            self.env.print()

            if status == ExecStatus.RETURN:
                self.env.pop()
                return (status, return_val)
        
        self.env.pop()
        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __call_func(self, call_node):
        func_name = call_node.get("name")
        called_by_var = False
        if func_name == "print":
            return self.__call_print(call_node)
        if func_name == "inputi":
            return self.__call_input(call_node)
        if func_name == "inputs":
            return self.__call_input(call_node)
        
        # print(f"func_name: {func_name}")
        # print(type(self.env.get(func_name)))
        if (self.env.get(func_name) is not None):
            if (type(self.env.get(func_name)).__name__ == "Element" and self.env.get(func_name).elem_type == "func"):
                func_name = self.env.get(func_name).get("name")
                called_by_var = True
            elif (type(self.env.get(func_name)).__name__ == "Element" and self.env.get(func_name).elem_type == "lambda"):
                print("here")
                return self.__call_lambda(call_node)
            else:
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Variable {func_name} does not hold a function.",
                )

        actual_args = call_node.get("args")
        print("actual_args: ")
        for arg in actual_args:
            print(arg)
        print()


        func_ast = self.__get_func_by_name(func_name, len(actual_args), called_by_var)
        formal_args = func_ast.get("args")

        print(f"***func_ast: {func_ast}")

        print("formal_args: ")
        for arg in formal_args:
            if(arg.elem_type == InterpreterBase.REFARG_DEF):
                print("is refarg")
            print(arg)
        print()

        if len(actual_args) != len(formal_args):
            if (called_by_var):
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Function {func_ast.get('name')} with {len(actual_args)} args not found",
                )
            super().error(
                ErrorType.NAME_ERROR,
                f"Function {func_ast.get('name')} with {len(actual_args)} args not found",
            )
        
        self.env.push()
        for formal_ast, actual_ast in zip(formal_args, actual_args):
            result = copy.deepcopy(self.__eval_expr(actual_ast))
            print("result: " +str(result.value()))
            arg_name = formal_ast.get("name")
            self.env.create(arg_name, result)
        _, return_val = self.__run_statements(func_ast.get("statements"))
        
        ref_args = {}
        for f_arg, a_arg in zip(formal_args, actual_args):
            if (f_arg.elem_type == InterpreterBase.REFARG_DEF): 
                print("**")
                print(f_arg.get("name"))
                print(a_arg.get("name"))
                print(self.env.get(a_arg.get("name")).value())
                if (f_arg.get("name") == a_arg.get("name")):
                    ref_args[f_arg.get("name")] = self.env.get(f_arg.get("name"))
                else:
                    self.env.set(a_arg.get("name"), self.env.get(f_arg.get("name")))
        
        self.env.pop()
        
        print("ref_args dict: " + str(ref_args))
        for key in ref_args.keys():
            print (ref_args[key].value())
            self.env.set(key, ref_args[key])
        
        return return_val
    
    def __do_lambda(self, call_node):
        return call_node
    
    def __call_lambda(self, call_node):
        print("lambduh")
        lambda_name = call_node.get("name")
        lambda_func = self.env.get(lambda_name)

        actual_args = call_node.get("args")
        print("actual_args: ")
        for arg in actual_args:
            print(arg)
        print()

        # func_ast = self.__get_func_by_name(func_name, len(actual_args), called_by_var)
        formal_args = lambda_func.get("args")

        print("formal_args: ")
        for arg in formal_args:
            if(arg.elem_type == InterpreterBase.REFARG_DEF):
                print("is refarg")
            print(arg)
        print()

        # if len(actual_args) != len(formal_args):
        #     if (called_by_var):
        #         super().error(
        #             ErrorType.TYPE_ERROR,
        #             f"Function {func_ast.get('name')} with {len(actual_args)} args not found",
        #         )
        #     super().error(
        #         ErrorType.NAME_ERROR,
        #         f"Function {func_ast.get('name')} with {len(actual_args)} args not found",
        #     )
        
        env = self.lambdas_to_env[lambda_name]
        print(f"env: {env.get_env()}")
        # env.print()
        # env = env.get_env()
        # print(env)
        # print()

        self.env.push_env(env.get_env())

        for formal_ast, actual_ast in zip(formal_args, actual_args):
            result = copy.deepcopy(self.__eval_expr(actual_ast))
            print("result: " +str(result.value()))
            arg_name = formal_ast.get("name")
            self.env.create(arg_name, result)
        _, return_val = self.__run_statements(lambda_func.get("statements"))
        
        ref_args = {}
        for f_arg, a_arg in zip(formal_args, actual_args):
            if (f_arg.elem_type == InterpreterBase.REFARG_DEF): 
                if (f_arg.get("name") == a_arg.get("name")):
                    ref_args[f_arg.get("name")] = self.env.get(f_arg.get("name"))
                else:
                    self.env.set(a_arg.get("name"), self.env.get(f_arg.get("name")))
        
        self.env.pop()
        
        print("ref_args dict: " + str(ref_args))
        for key in ref_args.keys():
            print (ref_args[key].value())
            self.env.set(key, ref_args[key])
        
        return return_val

    def __call_print(self, call_ast):
        output = ""
        for arg in call_ast.get("args"):
            result = self.__eval_expr(arg)  # result is a Value object
            output = output + get_printable(result)
        print("*printed by program*:")
        super().output(output)
        return Interpreter.NIL_VALUE

    def __call_input(self, call_ast):
        args = call_ast.get("args")
        if args is not None and len(args) == 1:
            result = self.__eval_expr(args[0])
            super().output(get_printable(result))
        elif args is not None and len(args) > 1:
            super().error(
                ErrorType.NAME_ERROR, "No inputi() function that takes > 1 parameter"
            )
        inp = super().get_input()
        if call_ast.get("name") == "inputi":
            return Value(Type.INT, int(inp))
        if call_ast.get("name") == "inputs":
            return Value(Type.STRING, inp)

    def __assign(self, assign_ast):
        var_name = assign_ast.get("name")
        value_obj = self.__eval_expr(assign_ast.get("expression"))
        if (type(value_obj).__name__ == "Element" and value_obj.elem_type == InterpreterBase.LAMBDA_DEF):
            print(f"assign_ast: {assign_ast}")
            self.lambda_name_to_ast[var_name] = value_obj
            self.lambdas_to_env[var_name] = copy.deepcopy(self.env)
            print("\ncheckin here: ")
            self.env.print()
            print(f"lambda_name_to_ast: {self.lambda_name_to_ast}")
            print(f"lambdas_to_env: {self.lambdas_to_env}")
        if (value_obj in self.lambda_name_to_ast):
            print("it's here!!!")
            print(f"env of {value_obj}: {self.lambdas_to_env[value_obj]}")
            self.lambdas_to_env[var_name] = self.lambdas_to_env[value_obj]
            value_obj = self.lambda_name_to_ast[value_obj]
        
        self.env.set(var_name, value_obj)

    def __eval_expr(self, expr_ast):
        print("here expr")
        print("type: " + str(expr_ast.elem_type))
        if expr_ast.elem_type == InterpreterBase.NIL_DEF:
            # print("getting as nil")
            return Interpreter.NIL_VALUE
        if expr_ast.elem_type == InterpreterBase.INT_DEF:
            return Value(Type.INT, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.STRING_DEF:
            # print("getting as str")
            return Value(Type.STRING, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.BOOL_DEF:
            return Value(Type.BOOL, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.VAR_DEF:
            var_name = expr_ast.get("name")
            val = self.env.get(var_name)
            print(f"val: {val}")
            if (type(val).__name__ == "Element" and val.elem_type == InterpreterBase.LAMBDA_DEF):
                print(f"yayuh, the var name is {var_name} and it has a lambduh")
                val = var_name
            if (var_name in self.func_name_to_ast):
                val = self.__get_func_by_name_for_var(var_name)
            if val is None:
                super().error(ErrorType.NAME_ERROR, f"Variable {var_name} not found")
            return val
        if expr_ast.elem_type == InterpreterBase.FCALL_DEF:
            return self.__call_func(expr_ast)
        if expr_ast.elem_type == InterpreterBase.LAMBDA_DEF:
            return self.__do_lambda(expr_ast)
        if expr_ast.elem_type in Interpreter.BIN_OPS:
            return self.__eval_op(expr_ast)
        if expr_ast.elem_type == Interpreter.NEG_DEF:
            return self.__eval_unary(expr_ast, Type.INT, lambda x: -1 * x)
        if expr_ast.elem_type == Interpreter.NOT_DEF:
            return self.__eval_unary(expr_ast, Type.BOOL, lambda x: not x)

    def __eval_op(self, arith_ast):
        left_value_obj = self.__eval_expr(arith_ast.get("op1"))
        right_value_obj = self.__eval_expr(arith_ast.get("op2"))

        # print("**")
        # print(left_value_obj)
        # print(right_value_obj)
        # print(type(left_value_obj).__name__ == "Element")

        # Create value objects if var is a function
        if (type(left_value_obj).__name__ != "Value"):
            left_value_obj = create_value(left_value_obj)
        if (type(right_value_obj).__name__ != "Value"):
            right_value_obj = create_value(right_value_obj)

        # print("**")
        # print(left_value_obj)
        # print(right_value_obj)

        if not self.__compatible_types(
            arith_ast.elem_type, left_value_obj, right_value_obj
        ):
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible types for {arith_ast.elem_type} operation",
            )
        
        if arith_ast.elem_type in ["==", "!=", "||", "&&", "!"]:
            if (left_value_obj.type() == Type.BOOL and right_value_obj.type() == Type.INT):
                if (right_value_obj.value() != 0):
                    right_value_obj = Value(Type.BOOL, True)
                else:
                    right_value_obj = Value(Type.BOOL, False)
            elif (left_value_obj.type() == Type.INT and right_value_obj.type() == Type.BOOL):
                print("**here")
                if (left_value_obj.value() != 0):
                    left_value_obj = Value(Type.BOOL, True)
                else:
                    left_value_obj = Value(Type.BOOL, False)
        
        elif arith_ast.elem_type in ["+", "-", "*", "/"]:
            if (left_value_obj.type() == Type.BOOL):
                if (left_value_obj.value()):
                    left_value_obj = Value(Type.INT, 1)
                else:
                    left_value_obj = Value(Type.INT, 0)
                
            if right_value_obj.type() == Type.BOOL:
                if (right_value_obj.value()):
                    right_value_obj = Value(Type.INT, 1)
                else:
                    right_value_obj = Value(Type.INT, 0)

        if arith_ast.elem_type not in self.op_to_lambda[left_value_obj.type()]:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible operator {arith_ast.elem_type} for type {left_value_obj.type()}",
            )   
                
        f = self.op_to_lambda[left_value_obj.type()][arith_ast.elem_type]
        # print("here eval")
        # print(arith_ast)
        # print("evaluating " + str(left_value_obj.type()) + " " + str(arith_ast.elem_type))
        # print("obj left: " + str(left_value_obj.value()))
        return f(left_value_obj, right_value_obj)

    def __compatible_types(self, oper, obj1, obj2):
        # DOCUMENT: allow comparisons ==/!= of anything against anything
        if oper in ["==", "!="]:
            return True
        if oper in ["+", "-", "*", "/", "&&", "||", "!"]:
            # if obj1.type() == Type.BOOL and obj2.type() == Type.BOOL:
            #     return True
            if obj1.type() == Type.BOOL or obj1.type() == Type.INT:
                if obj2.type() == Type.BOOL or obj2.type() == Type.INT:
                    return True
        return obj1.type() == obj2.type()

    def __eval_unary(self, arith_ast, t, f):
        value_obj = self.__eval_expr(arith_ast.get("op1"))
        if (arith_ast.elem_type == Interpreter.NOT_DEF):
            if (value_obj.type() == Type.INT):
                if (value_obj.value() != 0):
                    value_obj = Value(Type.BOOL, True)
                else:
                    value_obj = Value(Type.BOOL, False)
        if value_obj.type() != t:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible type for {arith_ast.elem_type} operation",
            )
        return Value(t, f(value_obj.value()))

    def __setup_ops(self):
        self.op_to_lambda = {}
        # set up operations on integers
        self.op_to_lambda[Type.INT] = {}
        self.op_to_lambda[Type.INT]["+"] = lambda x, y: Value(
            x.type(), x.value() + y.value()
        )
        self.op_to_lambda[Type.INT]["-"] = lambda x, y: Value(
            x.type(), x.value() - y.value()
        )
        self.op_to_lambda[Type.INT]["*"] = lambda x, y: Value(
            x.type(), x.value() * y.value()
        )
        self.op_to_lambda[Type.INT]["/"] = lambda x, y: Value(
            x.type(), x.value() // y.value()
        )
        self.op_to_lambda[Type.INT]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.INT]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )
        self.op_to_lambda[Type.INT]["<"] = lambda x, y: Value(
            Type.BOOL, x.value() < y.value()
        )
        self.op_to_lambda[Type.INT]["<="] = lambda x, y: Value(
            Type.BOOL, x.value() <= y.value()
        )
        self.op_to_lambda[Type.INT][">"] = lambda x, y: Value(
            Type.BOOL, x.value() > y.value()
        )
        self.op_to_lambda[Type.INT][">="] = lambda x, y: Value(
            Type.BOOL, x.value() >= y.value()
        )
        #  set up operations on strings
        self.op_to_lambda[Type.STRING] = {}
        self.op_to_lambda[Type.STRING]["+"] = lambda x, y: Value(
            x.type(), x.value() + y.value()
        )
        self.op_to_lambda[Type.STRING]["=="] = lambda x, y: Value(
            Type.BOOL, x.value() == y.value()
        )
        self.op_to_lambda[Type.STRING]["!="] = lambda x, y: Value(
            Type.BOOL, x.value() != y.value()
        )
        #  set up operations on bools
        self.op_to_lambda[Type.BOOL] = {}
        self.op_to_lambda[Type.BOOL]["&&"] = lambda x, y: Value(
            x.type(), x.value() and y.value()
        )
        self.op_to_lambda[Type.BOOL]["||"] = lambda x, y: Value(
            x.type(), x.value() or y.value()
        )
        self.op_to_lambda[Type.BOOL]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.BOOL]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

        #  set up operations on nil
        self.op_to_lambda[Type.NIL] = {}
        self.op_to_lambda[Type.NIL]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.NIL]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

        #  set up operations on funcs
        self.op_to_lambda[Type.FUNC] = {}
        self.op_to_lambda[Type.FUNC]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.FUNC]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

    def __do_if(self, if_ast):
        cond_ast = if_ast.get("condition")
        result = self.__eval_expr(cond_ast)
        print(f"result type: {result.type()}")
        if result.type() != Type.BOOL:
            if result.type() == Type.INT:
                if result.value() != 0:
                    result = Value(Type.BOOL, True)
                else:
                    result = Value(Type.BOOL, False)
            else:
                super().error(
                    ErrorType.TYPE_ERROR,
                    "Incompatible type for if condition",
                )
        
        if result.value():
            statements = if_ast.get("statements")
            status, return_val = self.__run_statements(statements)
            return (status, return_val)
        else:
            else_statements = if_ast.get("else_statements")
            if else_statements is not None:
                status, return_val = self.__run_statements(else_statements)
                return (status, return_val)

        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __do_while(self, while_ast):
        cond_ast = while_ast.get("condition")
        run_while = Interpreter.TRUE_VALUE
        while run_while.value():
            run_while = self.__eval_expr(cond_ast)
            print(f"run_while: {run_while.type()}")
            if run_while.type() != Type.BOOL:
                if run_while.type() == Type.INT:
                    print("true")
                    if run_while.value() != 0:
                        run_while = Value(Type.BOOL, True)
                    else:
                        run_while = Value(Type.BOOL, False)
                    print(f"post run_while: {run_while.type()}")
                else: 
                    super().error(
                        ErrorType.TYPE_ERROR,
                        "Incompatible type for while condition",
                    )
            if run_while.value():
                statements = while_ast.get("statements")
                status, return_val = self.__run_statements(statements)
                if status == ExecStatus.RETURN:
                    return status, return_val

        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __do_return(self, return_ast):
        expr_ast = return_ast.get("expression")
        if expr_ast is None:
            return (ExecStatus.RETURN, Interpreter.NIL_VALUE)
        value_obj = copy.deepcopy(self.__eval_expr(expr_ast))
        return (ExecStatus.RETURN, value_obj)
    
def main():
    #all programs will be provided to your interpreter as a python string,
    # just as shown here.

    program_source0 = """func main() {
        x = 5;
        x = -x + 5 + 9;
        y = true;
        y = !y;
        a = "sweetie ";
        b = "queen ";
        x = 5;

        print(x);
        print(y);
    }
    """
        
    program_source1 = """
    func foo(a){
        print(a);
    }
    
    func main() {
        y = true;
        x = main;
        foo(x, y);
        x(10);
        y(10);
    }
    """

    program_source2 = """
    func foo() { return bar; }

    func bar(a,b) { print(a,b); }

    func main() {
        a = foo();
        a(10,20);
    }
    """

    program_source2_5 = """

    func bar(){
        return nil;
    }

    func foo(){
        c = 10;
    }
    func main() {
        a = 5;
        foo();
        while (a == 5){
            b = 10;
            a = 6;
        }
        print(a);
        print(b);
        print(c);

    }
    """

    program_source3 = """
    func main() {
        a = -5;
        if (true == a){
            print("This will print!");
        }

        if (!0){
            print("This will print!");
        }

        if (false || 6){
            print("This prints!");
        }

        a = 3;
        while (a) {
            print("This prints 3 times!");
            a = a - 1;
        }

    }

    """

    program_source4 = """
    func foo() {
        b = 5;
        d = 4;
        f = lambda(a) { b = a*b; print(b); };
        return f;
    }

    func main() {
        x = foo();
        c = 20;
        x(c);
        z = 5;
        y = lambda(z) { print(z); };
        y(10);
    }
    """

    program_source5 = """
    func main() {
        x = foo;
        print(x == foo);
        x = foo();
        print(x == foo);
        x("bar");
    }

    func foo() {
        print("foo");
        return bar;
    }

    func bar(x){
        print(x);
    }
    """

    program_source6 = """
    func foo(a) {
        print(a);
    }

    func main() {
        a = foo;
        a(10);
    }
    """
    program_source7 = """
    func bar() {
        print("whoops");
        return;
    }

    func main() {
        val = nil;
        if (foo() == val){
            print("queen nil");
        }
    }

    func foo() {
        print("hello!");
        return 5;
    }
    """

    program_source8 = """
    func foo() {
        print("foo: ", a, " ", b, " ", c);
        b = b + 1;
    }
    func main(){
        a = 10;
        b = 20;
        bar();
        c = 70;
        print(c);
    }

    func bar(){
        print("bar: ", a, " ", b);
        c = 30;
        foo();
        print("main: ", a, " ", b);
    }
    """
    # this is how you use our parser to parse a valid Brewin program into 
    # an AST:

    i.run(program_source4)

i = Interpreter()
main()