def pretty_print(elem, indent=0, after_key=False):
    prefix = "  " * indent
    if isinstance(elem, dict):
        if after_key:
            print("{")
        else:
            print(prefix + "{")
        for k, v in elem.items():
            print(prefix + f"  {k}: ", end="")
            pretty_print(v, indent + 1, after_key=True)
        print(prefix + "}")
    elif isinstance(elem, list):
        if after_key:
            print("[")
        else:
            print(prefix + "[")
        for v in elem:
            pretty_print(v, indent + 1)
        print(prefix + "]")
    elif after_key:
        print(repr(elem))
    else:
        print(prefix + repr(elem))
