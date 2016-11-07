## -*- coding:utf-8 -*-
#coding=utf-8
from django.db import models
from django.http import HttpResponse,HttpResponseForbidden
from fields import RelatedForeignKey
from datetime import datetime
from django.contrib.auth.models import User

class AlertappLogin(models.Model):
    id          = models.AutoField('id',primary_key=True)
    user_id     = models.ForeignKey(User,verbose_name = '用户名',db_column='user_id', null=True, blank=True)
    first_login = models.DateTimeField('首次登录时间',blank=True,null=True,editable = False)
    last_login  = models.DateTimeField('最近登录时间',blank=True,null=True,editable = False)
    is_paid     = models.BooleanField("是否付费",default = False)
    time_length = models.IntegerField("试用天数",blank = True,null = True,default = 60)
# alter table alertapp_login add time_length int(4) not null default 60;

    class Meta:
        db_table = "alertapp_login"
        verbose_name = "app付费管理"
        verbose_name_plural = verbose_name
        ordering = ['-is_paid']



# Create your models here.
class Alert(models.Model):
    id            = models.AutoField('id',primary_key=True)
    t0_id         = models.ForeignKey('T0', verbose_name='T0编号', db_column='t0_id', null=True, blank=True)
    time          = models.DateTimeField('时间', null=False, blank=True, default=datetime.now)
    message       = models.CharField('内容', blank=True, max_length=765)
    normal        = models.BooleanField('正常报告', default=True)    # 0 alarm  1 normal

    display_index = 4

    class Meta:
        db_table = 'A1'
        verbose_name = '调压器上报信息'
        verbose_name_plural = '调压器上报信息纪录'
        ordering = ['-time']


class Alert_A2(models.Model):
    t0_id = models.ForeignKey('T0', verbose_name='T0编号', db_column='t0_id', null=True, blank=True)
    date  = models.DateField('报警日期', null=True, blank=True)

    display_index = 5

    def __unicode__(self):
        return u'[%s][%s]' % (str(self.date), self.t0_id)

    class Meta:
        db_table = 'A2'
        verbose_name = '调压器报警信息'
        verbose_name_plural = '调压器报警纪录'
        ordering = ['-date']


class Company(models.Model):
    name                = models.CharField('公司名称', blank=True, max_length=150)
    address             = models.CharField('公司地址', blank=True, max_length=150)
    city                = models.CharField('城市', blank=True, max_length=150)
    province            = models.CharField('省份', blank=True, max_length=60)
    postal_code         = models.CharField('邮政编码', blank=True, max_length=60)
    phone_num           = models.CharField('电话号码', blank=True, max_length=90)
    fax                 = models.CharField('传真号码', blank=True, max_length=90)
    pri_contact         = models.CharField('主要联系人', blank=True, max_length=150)
    legal_rep           = models.CharField('法人代表', blank=True, max_length=150)
    website             = models.CharField('网址', blank=True, max_length=150)
    email               = models.CharField('邮箱', blank=True, max_length=150)
    default_pay_term    = models.CharField('默认付款条件', blank=True, max_length=765)
    default_invoice_note = models.TextField('默认发票说明', blank=True)

    display_index = 0

    def __unicode__(self):
        return self.name

    class Meta:
        db_table = 'Company'
        verbose_name = '公司'
        verbose_name_plural = '公司信息表'
        ordering = ['id']

class P0(models.Model):
    t0_id            = models.ForeignKey('T0', verbose_name='T0编号', db_column='t0_id', null=True, blank=True)
    time             = models.DateTimeField('时间', null=True, blank=True)
    valve_pressure1  = models.IntegerField('出口压力', null=True, blank=True)
    high_pressure    = models.IntegerField('进口压力', null=True, blank=True)
    diff_pressure    = models.IntegerField('差压', null=True, blank=True)
    diff_pressure2   = models.IntegerField('差压2', null=True, blank=True)
    valve_pressure2  = models.IntegerField('进口压力2', null=True, blank=True)

    display_index = 6

    def __unicode__(self):
        return u"[%s] [%s] %s" % (str(self.time), self.t0_id, str(self.valve_pressure1), str(self.high_pressure),  str(self.diff_pressure), str(self.diff_pressure2))

    class Meta:
        db_table = 'P0'
        verbose_name = '压力'
        verbose_name_plural = '压力数据纪录'
        ordering = ['-time']

#虚拟表，用于原生sql提取压力数据结果集
class P0_result(models.Model):
    time            = models.DateTimeField('时间', blank=True, primary_key=True)
    valve_pressure1 = models.IntegerField('出口压力', null=True, blank=True)

class Station(models.Model):
    id          = models.AutoField('id', primary_key=True)
    company_id  = models.ForeignKey(Company, verbose_name='Company ID', db_column='Company ID', blank=True)
    name        = models.CharField('站点名称', blank=True, max_length=150)
    contact     = models.CharField('联系人', blank=True, max_length=150)
    chief       = models.CharField('站长', blank=True, max_length=150)
    address     = models.TextField('地址', blank=True)
    phone_num   = models.CharField('电话', blank=True, max_length=150)
    mobile      = models.CharField('联系人手机', blank=True, max_length=150)
    email       = models.CharField('电子邮件', blank=True, max_length=150)
    Memo        = models.TextField('备注', blank=True)

    display_index = 2

    def __unicode__(self):
        return self.name

    class Meta:
        db_table = 'Station'
        verbose_name = '输配分站'
        verbose_name_plural = '输配站信息表'
        ordering = ['id']

class P_Unit(models.Model):
    id      = models.IntegerField(primary_key=True)
    p_unit  = models.CharField(max_length = 10)

    def __unicode__(self):
        return self.p_unit

    class Meta:
        db_table = 'P_Unit'

class Gas_Type(models.Model):
    GT_id    = models.IntegerField(primary_key=True)
    Gas_Type = models.CharField(max_length = 24)

    def __unicode__(self):
        return self.Gas_Type

    class Meta:
        db_table = 'Gas_Type'


class T0(models.Model):
    _database = 'T0tables'
    id                 = models.AutoField('ID', primary_key=True)  
    station_id          = models.ForeignKey(Station, verbose_name='站点编号', db_column='station_id', null=True, blank=True, editable=False)
    device_id          = models.CharField('设备编号', blank=True, max_length=150, editable=False)
    name                = models.CharField('网点名称', blank=True, max_length=150,  editable=False)
    address             = models.CharField('网点地址', blank=True, null=True, max_length=150)
    latitude          = models.FloatField("纬度",blank = True,null = True)
    longitude          = models.FloatField("经度",blank = True,null = True)
    locat                 = models.CharField('locat', blank=True, max_length=20)
    repeater          = models.BooleanField('repeater', editable=False, default=0)
    time_choice     = ((0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(11,11),(12,12),(13,13),(14,14),
                   (15,15),(16,16),(17,17),(18,18),(19,19),(20,20),(21,21),(22,22),(23,23))
    report_time       = models.IntegerField('上报时间', blank=True, null = True, choices=time_choice, default = 2)
    interval_choice = ((1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9))
    check_interval  = models.IntegerField('调压器检测间隔', null=True, blank=True, choices=interval_choice, default = 1)
    report_interval_choice = ((1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(11,11),(12,12),(13,13),(14,14),
                              (15,15),(16,16),(17,17),(18,18),(19,19),(20,20))
    report_interval  = models.IntegerField('数据上报间隔', null=True, blank=True, choices=report_interval_choice, default = 1)
    alert_setting      = models.CharField('报警设置', null=True, blank=True, editable=True, max_length=50)
    center_ip           = models.CharField('中心地址', blank=True, null=True,max_length=50, editable=False)
    center_port       = models.CharField('中心端口', blank=True, null=True, max_length=50, editable=False)
    up_value1          = models.IntegerField('压力过低预警', null=True, blank=True, default=1500)            #压力下限
    up_value2          = models.IntegerField('放散压力预警', null=True, blank=True, default=2800)             #压力下限1
    up_value3          = models.IntegerField('主路切断压力', null=True, blank=True, default=3000)              #压力下限2
    up_value4          = models.IntegerField('副路切断压力', null=True, blank=True, default=3500)               #压力下限3
    sms_center        = models.CharField('短信中心', blank=True, null=True, max_length=50, editable=False )
    center_num       = models.CharField('数据中心号码', blank=True, null=True, max_length=50, editable=False )
    admin_num       = models.CharField('主控号码', blank=True, null=True, max_length=50, editable=False)
    alert_c         = models.CharField('报警1号码', blank=True, null=True, max_length=20)
    alert_d         = models.CharField('报警2号码', blank=True, null=True, max_length=20)
    alert_e         = models.CharField('报警3号码', blank=True, null=True, max_length=20)
    alert_f         = models.CharField('报警f号码', blank=True, null=True, max_length=20, editable=False)
    alert_g         = models.CharField('报警g号码', blank=True, null=True, max_length=20, editable=False)
    transmitter_range = models.IntegerField('变送器全量程', null=True, blank=True, default=5000, editable=False)
    flow_range            = models.IntegerField('流量全量程', null=True, blank=True, default=5000, editable=False)
    adjust                    = models.CharField('量程误差', null=True, blank=True, max_length=6, editable=False)
    h_range                 = models.IntegerField('中压变送器全量程', null=True, blank=True, default=5000, editable=False)
    d_range                 = models.IntegerField('差压变送器全量程', null=True, blank=True, default=5000, editable=False)
    p_unit                    = models.ForeignKey(P_Unit, verbose_name='出口压力单位',db_column='P_Unit_id', related_name='p_unit_id', blank=True, default=0)
    h_unit          = models.ForeignKey(P_Unit, verbose_name='进口压力单位', db_column='H_Unit_id', related_name='h_unit_id', blank=True, default=0)
    d_unit          = models.ForeignKey(P_Unit, verbose_name='差压单位', db_column='D_Unit_id', related_name='d_unit_id', blank=True, default=0)
    h_high          = models.IntegerField('中压上限', null=True, blank=True, default=5000)
    h_low           = models.IntegerField('中压下限', null=True, blank=True, default=0)
    d_high          = models.IntegerField('差压上限', null=True, blank=True, default=5000)
    d_low           = models.IntegerField('差压下限', null=True, blank=True, default=0) 
    is_artifacial_gas      = models.IntegerField('调压器气质', db_column='is_artifacial_gas',default=0)
    power_choice         = ((0,0),(1,1),(2,2),(3,3),(4,4),(5,5))
    power_state            = models.IntegerField('电源状态', null=True, blank=True, choices=power_choice, default = 5)
    work_state              = models.BooleanField('工作状态', default = True)       #设备安装状态
    tiaoyagui_model    = models.CharField('调压柜型号', blank=True, null=True, max_length=150)
    tiaoyaqi_model      = models.CharField('调压器型号', blank=True, null=True, max_length=150)
    entrance_pmeter   = models.CharField('进口压力表', blank=True, null=True, max_length=150,editable=False)
    exit_pmeter            = models.CharField('出口压力表', blank=True, null=True, max_length=150,editable=False)
    exit_presure            = models.FloatField('出口压力', null=True,  blank=True)
    inlet_pressure         = models.FloatField('进口压力', null=True,  blank=True)
    change_battery_time = models.DateField('更换电池时间', null=True, blank=True)
    modem              = models.IntegerField('modem', null=True, blank=True, editable=False)
    ZMD_count        = models.IntegerField('ZMD_count', null=True, blank=True, default=1, editable=True)
    new_id               = models.CharField('new_id', blank=True, max_length=150, editable=False)
    version               = models.IntegerField('version', null=True, blank=True, default = 91215, editable=True)
    other                  = models.TextField('其他',blank=True)
    #2010-06-01 changed by guo
    install_time          = models.DateField('安装时间', null=True, blank=True)
    manufactory        = models.CharField('manufactory', blank=True, max_length=20)
    structure              = models.CharField('structure', blank=True, max_length=10)
    use_type              = models.IntegerField('modem', max_length=1, null=True, blank=True, editable=False)
    show_device_id  = models.CharField('编号显示', blank=True, max_length=150, editable=False)

    display_index = 1
    def __unicode__(self):      
        try:
            return ("%s[%s]%s" % ((str(self.id)).zfill(2),(str(self.device_id))[3:], self.name.encode('utf-8'))).decode('utf-8')
        except:
            return "encode error"
    def get_absolute_url(self):
        return 'admin/polarwin/t0/%d' % self.id
    class Meta:
        db_table = 'T0'
        verbose_name = '调压器智能报警设备'
        verbose_name_plural = '报警设备设定'
        ordering = ['id']

class T0_result(models.Model):
    station_id = RelatedForeignKey(Station, verbose_name='站点编号', db_column='station_id', blank=True,primary_key=True)
    def __unicode__(self):
        return u"%s" % self.station_id.name

class T0_result_locat(models.Model):   
    label       = models.CharField('locat', blank=True, max_length=20,primary_key=True)
    isstation = models.IntegerField(null=True, blank=True, default=0)

    def __unicode__(self):
        return u"locat: %s" % self.label

class T3(models.Model):
    admin  = models.CharField('管理员', blank=True, max_length=150)
    passwd = models.CharField('密码', blank=True, max_length=150)

    def __unicode__(self):
        return u'admin: %s' % self.admin

    class Meta:
        db_table = 'T3'

class TbUpdateLog(models.Model):
    tb_name       = models.CharField(max_length=150)
    row_id          = models.IntegerField()
    clum_name = models.CharField(max_length=25)

    display_index = 8

    def __unicode__(self):
        return u"Table %s, row %s" % (self.tb_name, str(self.row_id))

    class Meta:
        db_table = 'tb_update_log'

class ReportAllDetail(models.Model):
    device_id         = models.CharField(max_length=150, primary_key=True)
    name               = models.CharField(max_length=150)
    avgpressure    = models.DecimalField(max_digits=19, decimal_places=10)
    maxpressure   = models.IntegerField()
    minpressure    = models.IntegerField()

    def __unicode__(self):
        return u"device_id %s" % self.device_id

class ReportAlarmDetail(models.Model):
    id                         = models.IntegerField(max_length=50,default = 1,primary_key=True)
    device_id            = models.CharField(max_length=150)
    name                  = models.CharField(max_length=150)
    ntime                   = models.CharField('时间',max_length=150, null=True, blank=True)
    normalmessage = models.CharField('恢复内容',null=True, blank=True, max_length=765)
    atime                  = models.CharField('时间',max_length=150, null=True, blank=True)
    alarmmessage   = models.CharField('报警内容', blank=True, max_length=765, default='')

    def __unicode__(self):
        return u"device_id %s" % self.device_id

class StationAlarmDetail(models.Model):
    ZMD_count       = models.IntegerField()
    device_id       = models.CharField(max_length=150, primary_key=True)
    name            = models.CharField(max_length=150)
    address         = models.CharField(max_length=150)
    tiaoyaqi_model  = models.CharField(max_length=150)
    alarmnum        = models.IntegerField()

    def __unicode__(self):
        return u"device_id %s" % self.device_id


class Auth_user_stations(models.Model):
    id         = models.AutoField('id', primary_key=True)
    user_id    = models.ForeignKey(User, db_column='user_id', null=True, blank=True)
    station_id = models.ForeignKey(Station, verbose_name='station_id', db_column='station_id', null=True, blank=True)

    def __unicode__(self):
        return u"user: %s, station: %s" % (self.user_id.username, self.station_id.name)

    class Meta:
        db_table = 'auth_user_stations'
        verbose_name = '用户站点对应表'
        verbose_name_plural = '用户站点对应设定'

class UserT0column(models.Model):
    id       = models.AutoField('id', primary_key=True)
    userid   = models.IntegerField()
    t0column = models.TextField(max_length=400,null=True, blank=True)

    def __unicode__(self):
        return u"userid: %s, t0column: %s" % (self.userid, self.t0column)

    class Meta:
        db_table = 'user_t0column'
        verbose_name = '用户T0列对应表'

class Same_id(models.Model):
    id            = models.AutoField('id', primary_key=True)
    device_id     = models.CharField('设备编号', blank=True, max_length=20, editable=False)
    station       = models.CharField('站点', blank=True, max_length=20)
    t0_device_id  = models.CharField('t0设备编号', blank=True, max_length=20, editable=False)
   
    def __unicode__(self):
        return u"%s" % (self.device_id)

    class Meta:
        db_table = 'Same_id'
        verbose_name = '相同id表'

class Auth_user(models.Model):
    id        = models.AutoField('id', primary_key=True)
    username  = models.CharField('用户名称', blank=True, max_length=30, editable=False)

    def __unicode__(self):
        return u"%s" % (self.username)

    class Meta:
        db_table = 'auth_user'
        verbose_name = '用户表'

class Block_alarm(models.Model):
    id             = models.AutoField('id', primary_key=True)
    t0_id          = models.ForeignKey('T0', verbose_name='T0编号', db_column='t0_id', null=True, blank=True)
    ZMD_count      = models.IntegerField('ZMD_count', null=True, blank=True, default=1, editable=True)
    set_time       = models.DateTimeField('时间', null=False, blank=True, default=datetime.now)
    start_time     = models.DateTimeField('开始时间', null=True, blank=True)
    end_time       = models.DateTimeField('结束时间', null=True, blank=True)
    is_executed    = models.BooleanField('是否已修改', editable=False, default=0)
    is_quit        = models.BooleanField('是否遮蔽', editable=False, default=0)
    user_name      = models.CharField('修改用户', blank=True, max_length=20, editable=False)

    def __unicode__(self):
        return u"%s" % (self.t0_id)

    class Meta:
        db_table = 'block_alarm'
        verbose_name = '报警设置表'
        ordering = ['-set_time']

class Message_telephone(models.Model):
    id              = models.AutoField('id', primary_key=True)
    telephoneNumber = models.CharField('手机', blank=True, max_length=11, editable=False)
    name            = models.CharField('联系人', blank=True, max_length=20, editable=False)
    is_inform       = models.BooleanField('是否通知', default = True)
    information     = models.CharField('备注', blank=True, max_length=28, editable=False)

    def __unicode__(self):
        return u"%s" % (self.telephone)

    class Meta:
        db_table = 'message_telephone'
        verbose_name = '联系人手机表'

class Message_message(models.Model):
    id           = models.AutoField('id', primary_key=True)
    name         = models.TextField('短信记录', blank=True, max_length=100, editable=False)
    date         = models.DateTimeField('记录时间', null=True, blank=True)
    status       = models.BooleanField('短信状态', default = True)
    is_sent      = models.BooleanField('是否发送', default = True)
    station      = models.CharField('站点', blank=True, max_length=20, editable=False)

    def __unicode__(self):
        return u"%s" % (self.contetn)

    class Meta:
        db_table = 'message_message'
        verbose_name = '短信历史表'
        ordering = ['-date']

class Station_telephone(models.Model):
    id           = models.AutoField('id', primary_key=True)
    telephone_id = models.ForeignKey('Message_telephone', verbose_name='联系人id', db_column='telephone_id', null=True, blank=True)
    station      = models.CharField('站点', blank=True, max_length=45, editable=False)

    def __unicode__(self):
        return u"%s" % (self.station)

    class Meta:
        db_table = 'station_telephone'
        verbose_name = '站点手机表'

class Message_message_telephones(models.Model):
    id           = models.AutoField('id', primary_key=True)
    telephone_id = models.ForeignKey('Message_telephone', verbose_name='联系人id', db_column='telephone_id', null=True, blank=True)
    message_id   = models.ForeignKey('Message_message', verbose_name='短信id', db_column='message_id', null=True, blank=True)

    def __unicode__(self):
        return u"%s" % (self.telepone_id)

    class Meta:
        db_table = 'message_message_telephones'
        verbose_name = '短信联系人表'

class Message_model(models.Model):
    id       = models.AutoField('id', primary_key=True)
    name     = models.CharField('名称', blank=True, max_length=20, editable=False)
    content  = models.CharField('内容', blank=True, max_length=100, editable=False)

    def __unicode__(self):
        return u"%s" % (self.name)

    class Meta:
        db_table = 'message_model'
        verbose_name = '模板表'

class Authority(models.Model):
    id         = models.AutoField('id', primary_key=True)
    use_id     = models.IntegerField('用户编号', null=True, blank=True, default=1, editable=True)
    close_page = models.CharField('遮蔽页面', blank=True, max_length=45, editable=False)

    def __unicode__(self):
        return u"%s" % (self.use_id)

    class Meta:
        db_table = 'authority'

        verbose_name = '用户权限表'

class Addgroup(models.Model):
    id         = models.AutoField('id', primary_key=True)
    device_id  = models.CharField('设备编号', blank=True, max_length=10, editable=False)
    type       = models.IntegerField('类型', null=True, blank=True, default=1, editable=True)
    group      = models.CharField('组名', blank=True, max_length=45, editable=False)

    def __unicode__(self):
        return u"%s" % (self.device_id)

    class Meta:
        db_table = 'add_group'
        verbose_name = '添加组表'

class Showrow(models.Model):
    id         = models.AutoField('id', primary_key=True)
    show_id    = models.CharField('显示编号', blank=True, max_length=100, editable=False)
    use_id     = models.IntegerField('用户编号', null=True, blank=True, default=1, editable=True)

    def __unicode__(self):
        return u"%s" % (self.use_id)

    class Meta:
        db_table = 'Show_row'
        verbose_name = '显示列名表'
