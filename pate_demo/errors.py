weak_classes_empty_error = "For empty args.weak_classes set the parameter to: '' (empty string)."
perfect_balance_type_error = 'Perfect balance type requires empty weak_classes.' + ' ' + weak_classes_empty_error


def check_perfect_balance_type(args):
    if args.weak_classes != []:
        raise Exception(perfect_balance_type_error)
