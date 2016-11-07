# -*- coding:utf-8 -*-

from django.db import models

class RelatedManyToManyField(models.ManyToManyField):
    
    def __init__(self, *args, **kwds):
        #self.related_set_name = kwds.pop('related_set_name')
        self.__manipulator = None

        models.ManyToManyField.__init__(self, *args, **kwds)

    def get_manipulator_fields(self, opts, manipulator, change,
        
        name_prefix='', rel=False, follow=True):
        
        self.__manipulator = manipulator

        return super(RelatedManyToManyField,self).get_manipulator_fields(
            opts, manipulator, change, name_prefix, rel, follow)

    def get_choices_default(self):
        choices = []
        from psrv.padmin.models import Station
        if hasattr(self.__manipulator, 'original_object'):
            pd = self.__manipulator.model.objects.get(id=self.__manipulator.original_object.id)
            sts = Station.objects.extra(where=["`Company ID` = %s" % pd.company.id])
            choices = [(st.id, str(st)) for st in sts]
            pass
        else:
            sts = Station.objects.all()
            choices = [(st.id, str(st)) for st in sts]            
        self.__manipulator = None
        return choices

class RelatedForeignKey(models.ForeignKey):
    
    def __init__(self, *args, **kwds):
        #self.related_set_name = kwds.pop('related_set_name')
        self.__manipulator = None

        models.ForeignKey.__init__(self, *args, **kwds)

    def get_manipulator_fields(self, opts, manipulator, change,
        
        name_prefix='', rel=False, follow=True):
        
        self.__manipulator = manipulator

        return super(RelatedForeignKey,self).get_manipulator_fields(
            opts, manipulator, change, name_prefix, rel, follow)

    def get_choices_default(self):

        from psrv.padmin.models import Station
        choices = []
        if hasattr(self.__manipulator, 'original_object'):
            cmpid    = self.__manipulator.original_object.station_id.company_id.id
            sts          = Station.objects.extra(where=["`Company ID` = %s" % cmpid])
            choices  = [(st.id, str(st)) for st in sts]
        else:
            sts         = Station.objects.all()
            choices = [(st.id, str(st)) for st in sts]
        self.__manipulator = None
        return choices

