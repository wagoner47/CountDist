from __future__ import print_function
from configobj import ConfigObj


class MyConfigObj(ConfigObj):
    def __init__(self, infile=None, options=None, configspec=None,
                 encoding=None, interpolation=True, raise_errors=False,
                 list_values=True, create_empty=False, file_error=False,
                 stringify=True, indent_type=None, default_encoding=None,
                 unrepr=False, write_empty_values=False, _inspec=False):
        super(MyConfigObj, self).__init__(infile, options, configspec, encoding,
                                          interpolation, raise_errors,
                                          list_values, create_empty, file_error,
                                          stringify, indent_type,
                                          default_encoding, unrepr,
                                          write_empty_values, _inspec)
    
    def _write_line(self, indent_string, entry, this_entry, comment):
        if not self.unrepr:
            val = super(MyConfigObj, self)._decode_element(
                super(MyConfigObj, self)._quote(this_entry))
        else:
            val = repr(this_entry)
        
        return "%s%s%s%s%s" % (indent_string,
                               super(MyConfigObj, self)._decode_element(
                                   super(MyConfigObj, self)._quote(entry,
                                                                   multiline=False)),
                               super(MyConfigObj, self)._a_to_u("= "), val,
                               super(MyConfigObj, self)._decode_element(
                                   comment))
