"""describe class"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod

"""___Third-Party Modules___"""
# import here

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "11/09/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class IMainSimulationsWidget(ABC):
    @abstractmethod
    def set_export_button_disabled(self, disabled: bool):
        pass


metadic = {}


def _generatemetaclass(bases, metas, priority):
    trivial = lambda m: sum([issubclass(M, m) for M in metas], m is type)
    # hackish!! m is trivial if it is 'type' or, in the case explicit
    # metaclasses are given, if it is a superclass of at least one of them
    metabs = tuple([mb for mb in map(type, bases) if not trivial(mb)])
    metabases = (metabs + metas, metas + metabs)[priority]
    if metabases in metadic:  # already generated metaclass
        return metadic[metabases]
    elif not metabases:  # trivial metabase
        meta = type
    elif len(metabases) == 1:  # single metabase
        meta = metabases[0]
    else:  # multiple metabases
        metaname = "_" + "".join([m.__name__ for m in metabases])
        meta = noconflict_makecls()(metaname, metabases, {})
    return metadic.setdefault(metabases, meta)


def noconflict_makecls(*metas, **options):
    """Class factory avoiding metatype conflicts. The invocation syntax is
    makecls(M1,M2,..,priority=1)(name,bases,dic). If the base classes have
    metaclasses conflicting within themselves or with the given metaclasses,
    it automatically generates a compatible metaclass and instantiate it.
    If priority is True, the given metaclasses have priority over the
    bases' metaclasses"""

    priority = options.get("priority", False)  # default, no priority
    return lambda n, b, d: _generatemetaclass(b, metas, priority)(n, b, d)
