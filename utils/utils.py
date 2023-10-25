from pydoc import locate
import inspect


def argparse_summary(arg_list, parser):
    arg_dict = vars(arg_list)
    action_groups_dict = {}
    for i in range(len(parser._action_groups)):
        action_groups_dict[parser._action_groups[i].title]=[]
    for j in parser._actions:
        if j.dest == "help":
            continue
        try:
            action_groups_dict[j.container.title].append((j.dest, arg_dict[j.dest]))
        except:
            print(f"not working: {j.dest}")

    value = "########################ArgParseSummaryStart########################"
    len_group_var = 55
    for k in parser._action_groups:
        group = k.title
        length_filler = len_group_var-len(group)
        length_filler1 = length_filler-(length_filler//2)
        length_filler2 = length_filler-length_filler1
        value+= f"\n{''.join(['-']*length_filler1)}{group}{''.join(['-']*length_filler2)}"
        for l in action_groups_dict[group]:
            value += "\n  {0:<25s}: {1:21s}  ".format(l[0], str(l[1]))
    value += "\n########################ArgParseSummaryEnd########################"
    print(value)


def get_argparser_group(title, parser):
    for group in parser._action_groups:
        if title == group.title:
            return group
    return None


def get_class_by_path(dot_path=None):
    if dot_path:
        MyClass = locate(dot_path)
        assert inspect.isclass(MyClass), f"Could not load {dot_path}"
        return MyClass
    else:
        return None
