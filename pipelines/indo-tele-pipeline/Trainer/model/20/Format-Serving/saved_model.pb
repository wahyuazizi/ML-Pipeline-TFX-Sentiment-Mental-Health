иЇ
ю-С-
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
Ё
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	
Ј
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 

ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
Г
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
Ѕ
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitintџџџџџџџџџ

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.12v2.10.0-76-gfdfc646704c8
ДK
ConstConst*
_output_shapes	
:Н	*
dtype0*љJ
valueяJBьJН	ByaBinternetBminBgaBkakBbangetBajaBygBpakeBjaringanBdmBkuotaBkaloBgakBnyaBgangguanBbgtBlemotBsemangatBpakaiBnomorBkasihBgueByukBudahBtolongBsihBlayananBcekBbenerBwifiBterimaBnihBnontonBbikinBtpB	telkomselBjamBiniBviaBtemanB	pelangganBonlineBmatiBkoneksiBhaiBsampeBpagiBmuluBloBlancarBkaliBidBgbBerrorBemangBdehBdahBcobaBsinyalBmohonBmalemBkerjaanBkendalaBjelekBjdBgmnBgabisaBdrBdownBdgnBdataBuntungBtagihanBsiangBsetiaBseninBrumahBonBnyamanBnormalBmiminBmalamBmaafBluBlagiBkerjaBkakakBilangBhpBhaloBgwBgituBgimanaBgantiBgBbisaBbiarBayangByoutubeBwoiBwkwkBwBudhBtksBtheBterkaitBtauBstabilBseharianBrestartBproviderBprivasiBpenggunaBpasBparahBnggaBmyBmingguBlohBlhoBlagBkuBkerjanyaBkabelBjgB
internetanBinfoinBhariBguysBginiBdapetBcariBberesBbeBbarengBbantuBapaBamanBaksesByourByaaBupgradeBtuhBtlgBtibaBtetapBterimakasihBtemenBtanggalBsumpahBsulisBsukaBsudahB	streamingBsoreBsobatBsmooaBsmBskrgBsemalemBsehatBseginiB
secepatnyaBsayaBrtoBprodukBpaketBnyalaBnoBnggakBmerahBmakasihBmakanBlumayanBliveBlgBleletBlangsungBlamaB
kompensasiBkoBknpB
kendalanyaBkenapaBkemarenBkapanBkalauBkadangBkacauBjdiBiyaBinternetnyaBinformasikanBinfoBikutanBgoogleBgangguBerorBdyBdongBdigitalBdaerahBcerahBcepetBbyBblmB
berkendalaBberbagiBbeliBbagusBaplikasiBallahB	aktivitasByakByahByaallahBwirelessBwibBwfhBwarungBvideoBurgentBupdateBunregBtwitterBtutorialBtugasBtotalBtoBtipeBtinggalBtiaptiapBteriakB	terdaftarBterbaikBtemaniBtelponBtelkomBteknisiBtayanganB	tanggapanBsuperBstudyBspesialBspeedBsisBsinyalnyaaaB	sinyalnyaBsilakanBsiiiBsiiB	signalnyaBsignalBserasaB	sepuasnyaBselfBselasaBselamatBsehariBsdhBsayangBsatsetsatsetBsaranBsalahBsahabatBrugiBringanBribetBrianBrendiBpulsaBpudingBproblemBpoBplissssBpingBpihakBpertanggungjawabanBperbaikiB	perbaikanBpenyesuaianBpengenB
pendidikanBpasangBokelahBofflineBnyuruhBnuuuBniBngeluhBngelekBngelagBmudahBmodemnyaB	menikmatiBmendungBmeetingBmauBmaskerBmanualBmantepppBmalmingBmainBmahalBlupaBlucuBlossBlosBlolaBlokasiBlihatBliatBlgiBlatihBlaporanBkuatBkualitasBkoutaBkotaBkontenB
koneksinyaBkonekBkompensasinyaBkhawatirB	kesehatanBkerenBkerasaBkemarinBkeluargaBkelBkejadianBkecB	kebutuhanBkayakBkayaBkarenaBkamuBkBjumlahBjugaBjelasBjayaBjagoBjagaBiyaaBinfokanBindahnyaBinBiBheheheBhealingBhangusBhalloBgtuBgtBgiliranBgemoyBgedeBgbsBgapapaBgamingBgameBgaesBfullBfollowBfeedbackBdulBdrakorBdorongBdoBdirumahB
diperbaikiB
dipeliharaBdipakeB
dilaporkanB
dibutuhkanB	diberikanB
diaktifkanBdariBdalamBcuacaBcsBconnectBcaranyaBcapekBcapeBbukaBbrpBboothnyaBbisnisBbestB
bermasalahB	berjamjamBberhariBbenerinBbebBbayarB	batasanmuBbanyakB
bantuannyaBbakatBbahkanBbagibagiB	apresiasiBantiBanjirBanjingBalhamdulillahBadminBzamanByuukkByowlooByoutubepodcastsampaiB
youtubenyaByooooByoByangByakokByaampunByaaaBxixiiBwuzhwuzhBwoyBwowBwoooiBwkwkwBwilkBwilayahBwebsiteBwebBwawanBwaipaiBwadahBwaBvisionBviolentBvidioB	valentineB	vaksinasiBuserBusahakuBupnormalBuploadBupayaBupB	unlimitedBudahlahBudaBtwBtvBtungguBtumbenBtukanBtugassssBtuBttBtrustedBtrusBtrsBtroubelBtrimaBtransformasiBtpiBtoyaBtobatBtmenBtitikBtindakBtiktokanBtiktokBtidurBthrotleBthanksBthBtetepBtestiBterusBterulangBterukurBtersebutB	tersayangBterputusB	terpantauBterpaksaBternyataBterluasBterjagaBtercoverB	terbilangBterbengkalaiBterbaruBtenangBtemporerB
tementemenBtelatBtegelnyaBteamBtdkBtdiBtarifBtapiBtanpeBtanpaB	tangerangB	tanboykunBtanahairBtalentaBtakutBtailahBtaiBtaekBtaaaiiiBsyBsusahhhBsusahBsurviveBsupportBsumpahhhBsumberBsulitBsukakBsucksBstopBsteffychibiBstatusB	speedtestBsoundBsosmedBsolusiBsodBsoaleBsmpBsmartBskrngBskilldanBskgBsisaBsinyalsinyalBsintingggggBsinisukaBsimonBsimBsilangBsiapinBsialBsiB	shopeepayBshitBsgtBsfBsetresB	setiabudiBsetelahBsetauBsetannBsesuaiBservicesBserviceBserverBserbaBseputarBsepiB
sepertinyaBsepatanBsentuhBsentraBsengajaBsenengBsendiriBsenayanBsemuaBsemogaBsemnjakBseminarB	semiingguBseluruhBselemotBselatanBselaluBsekolahBsekarangkitaBsekarangBsejumlahB
sejenisnyaBsegeraBsegarBsedangB
sebenernyaBsebelB	sebandingB	searchingBsaruaBsampBsamaanBsalevelBsalamBsakitBsajaBsadarBrusakBrupiahBroomBrobackB	rewardnyaBrevisianBrestoredBresponBresepBrepotBrendetBrelasiBrekomendasiBrekanB
registrasiBrefreshBreferralBrbBrameBramBraizelBquarterBqrisBputusBputerBpusatBpulsanyaBpulakBpuasBprotesBprosesBpromoB
programnyaBprogramB	prismediaBprimerBprhBpremiumBpotensiBponBplnBplisBpkeBpisannBpilihB	pettersonBpesenB
perusahaanB	perubahanBpersibB
perpanjangBpermasalahanB	perhatianBpercayaBperbulanBperbaikannyaB	perangkatBpenyelesaiannyaB
penyelamatBpenyebabnyaBpenuhBpenguranganBpengimputanBpengenxBpengembanganB
pengecekanB	pengajianB
penelitianB
pemotonganBpemilikBpelosokBpeliharaBpelayanannyaBpelanggannyaBpelBpdhlBpdhalBpdamBpastinyaBpandemiBpamulangBpalingBpagimuBpacketBovoBoutsideBoutageBotakBorbitBokeBoiiiiBofficeBoffBodeBnyiksaBnyeselBnyepiBnyebelinBnyarinyaBnyariBnyambungBnyalainBnyBnugasB
notifikasiBnothingBnotedB	nongkrongBnonBnomornyaBnikmatiBnikmanBniiiBnichBniceBniatBngomongBngepetBngemilBngelegBngecekBngapaBngambekBngadiBngadattBnetizenB	netflixanBnetflixBnerimaBnemeninBnehBnegatifBneedBnapasiBnapaBnanyaB	nanggepinBnamaBnBmwBmutualBmuterBmusuhanBmusikBmusicBmurattalBmurahBmumpungB
multimediaBmuBmotogpBmomongBmodemBmoBmlmBmintBmingguanBmilanBmieBmiBmfB	merugikanBmeregistrasiB	merasakanBmerBmenuaBmentokBmentabBmenjagaBmenitBmengupayakanBmengujiBmengonfirmasiBmenginformasikanBmenghubungiB
menghitungB
menghilangBmengenaiBmengeluhkanBmengeluhBmengecekB	mengaksesBmencariBmenarikBmenangBmembantuBmemakaiBmelaluiBmelajuBmbkBmbakB
masyarakatBmasukBmasiBmasakanBmasakBmasBmarahBmantapsB
mangajukanB	mandalikaBmanapunBmanaBmampangpondokBmaksimalBmakeBmakanyaBmakannyaBmaintenancenyaBmahBmagerBlwBlurdB	luncurkanBlunasB
lumayankanBluarBlsngBlolosBlogoBlivingBlinkBlicikBletBlengkapBlemottBlemotnyaBlemburBlekBlehBlegaBldrBlarisB	laptopnyaBlanjutinBlanjutiBlangkahB	langgananBlancarkemampuanBlampuBlambanBlakukanBlakBlainBlahyaaBlahhhBlaguBlagiiiiiBkykBkunjungiBkuliahB	krumahnyaBkrnBkoordinatnyaBkontakBkonsumenBkoneksinyaaBkoneksiiBkondisiBkokBkodeBkntlllBkntlBkngnBknapaBkloBkktBkitaBkirainBkiraBketemuBkeseruannyaBkeselBkeseharianmuBkesalB	kesabaranBkerjasamanyaBkepentalBkencangBkenapeBkenapasihhhhBkenapasiBkenaBkembaliBkemanaBkemampuanningkatinBkelipBkelapBkekB	kehabisanBkeceBkebutuhanbuatBkeabisanBkayanyaBkataB	kasihrikoBkartuBkarnaBkampretBkakkkkkBkahBkagetBkagaBkaburBjurnalisBjuancokBjtBjlekBjjBjiwaBjgnBjgaBjedaBjebolBjaringannyaBjarangBjapriBjalurBjadwalBjadiinBistriB	istighfarBislamiBiskanBinternetnyaaaBinternasionalnyaBintelBinsideBiniiB	ingikutinB
informatifBinetBindustriB	indonesiaBindihomBindiBimpulsesBidupBidiotBidaBhungkulBhuftB	hubungkanBhubungiBhubunganBhotstarBhororBhomeBhmmmBhmmBhiksBhihihiBhihiBhiBhematBhatikuBhatiBhasilnyaBhandalBhampirBhalooBhallooBhahaBhahBgueeeBguaBgpunyaBgotBgopayBgkBgintingBgimnBgilsBgeusBgercepBgemesBgelarBgegaraBgbblBganjangganjingBgandengB
gampiiilllBgaleriBgajelassBgaisBgadaBgaadaBfuvkBfupBfuckingBforBfixBfiturB	finansialBfinallyBfinalBfetishBfavoriteBfastBfaseBfakkkBevenBeusinaBeuiiBestimasiBeraBenakBemosiBehBeBduweBduluuuBdtgBdriveBdritadiBdriBdownloadBdokterBdoangBdllBdivingBditindakBdisuruhBdisneyBdiselesaikanBdisaatB	dirumahkuBdirujakB	direstuinB	direfreshB	diragukanB	diperiksaBdipakaiBdimohonB	dikerjainBdikasihB
dijanjikanBdihujatB
digangguinBdieBdidapatBdibikinBdibawaBdibantuBdiaksesBdetikB	detailnyaBdetailBdemandBdedikasiBdebutBdeboraBdapatBdanaBdampakBdahkBdaftarBdBcuunggggBcustomBcustBcurcolBcumanBcumaBcuanBcreatorBcowoBcostumerBcopotBcontohBcontentB
connectionBcomplainBcoklatBcobainBcmiiwBcityBciputatBcintamuBceritainBcepatBcemilanBceleronBcelanaBcekvnomornyaBcekinBcareBcardBcaptureBcantBcamilanBcakeppBcafeBcabutBbyuBbykBbutuhBbutBbusukBbungkusBbukB	bufferingBbucinBbuatBbskBbsBbroBbraderBboxBbosenBbosBborneoBboosterBbnykBbnrBblokirB
blbonusnyaBbiyarBbisaaBbillingBbiasanyaBbiasaBbgttBbgsttttBberupayaBbertanyaBbertahanB	bersyukurBberlanggananB
berkembangBberkedipBberisikontenBberhentiBberhasilBberharihariBberharapBberdoaBberdasarkanB	berdampakBberbedaBberatBberaktfitasB
berakhiranBberadaptasiBberadaBbenefitBbendaBbenciBbelomBbelaBbedroomBbebasBbayiBbatinnyaBbatasBbangurBbangunanBbangkeBbangeettBbangeeeetttttB
balikpapanBbaliBbaikBbahkaBbaeeeeeBbadanBbackupanBbabikB	awakeningBatasBasyikBastagaaBastagaBasikBasahBarahinBappsBapapunBapalagiBanywayBantriBanthonyBanjngggggggBanjngBanjjBanjgBanehBandriBandBanakBanBaminBambilBaloneBalinBalesanB	alasannyaBalamatBalaBakuuuuuBaktifBakanBajggggBajaaBagarBafriadiBaeBadeBadaBacBabsenBabisBabang
ОL
Const_1Const*
_output_shapes	
:Н	*
dtype0	*L
valueїKBєK	Н	"шK                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~                                                                                                                                                                                                                                                      Ё       Ђ       Ѓ       Є       Ѕ       І       Ї       Ј       Љ       Њ       Ћ       Ќ       ­       Ў       Џ       А       Б       В       Г       Д       Е       Ж       З       И       Й       К       Л       М       Н       О       П       Р       С       Т       У       Ф       Х       Ц       Ч       Ш       Щ       Ъ       Ы       Ь       Э       Ю       Я       а       б       в       г       д       е       ж       з       и       й       к       л       м       н       о       п       р       с       т       у       ф       х       ц       ч       ш       щ       ъ       ы       ь       э       ю       я       №       ё       ђ       ѓ       є       ѕ       і       ї       ј       љ       њ       ћ       ќ       §       ў       џ                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~                                                                                                                                                                                                                   Ё      Ђ      Ѓ      Є      Ѕ      І      Ї      Ј      Љ      Њ      Ћ      Ќ      ­      Ў      Џ      А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      №      ё      ђ      ѓ      є      ѕ      і      ї      ј      љ      њ      ћ      ќ      §      ў      џ                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~                                                                                                                                                                                                                   Ё      Ђ      Ѓ      Є      Ѕ      І      Ї      Ј      Љ      Њ      Ћ      Ќ      ­      Ў      Џ      А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      №      ё      ђ      ѓ      є      ѕ      і      ї      ј      љ      њ      ћ      ќ      §      ў      џ                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~                                                                                                                                                                                                                   Ё      Ђ      Ѓ      Є      Ѕ      І      Ї      Ј      Љ      Њ      Ћ      Ќ      ­      Ў      Џ      А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      №      ё      ђ      ѓ      є      ѕ      і      ї      ј      љ      њ      ћ      ќ      §      ў      џ                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~                                                                                                                                                                                                                   Ё      Ђ      Ѓ      Є      Ѕ      І      Ї      Ј      Љ      Њ      Ћ      Ќ      ­      Ў      Џ      А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
H
Const_4Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_5Const*
_output_shapes
: *
dtype0	*
value	B	 R

Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/v
y
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes
:*
dtype0

Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_14/kernel/v

*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes

: *
dtype0

Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
: *
dtype0

Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_13/kernel/v

*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes

:@ *
dtype0

Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_12/bias/v
y
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_12/kernel/v

*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes

:@*
dtype0

Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	N*,
shared_nameAdam/embedding/embeddings/v

/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v*
_output_shapes
:	N*
dtype0

Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/m
y
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes
:*
dtype0

Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_14/kernel/m

*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes

: *
dtype0

Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
: *
dtype0

Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_13/kernel/m

*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes

:@ *
dtype0

Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_12/bias/m
y
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_12/kernel/m

*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes

:@*
dtype0

Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	N*,
shared_nameAdam/embedding/embeddings/m

/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m*
_output_shapes
:	N*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0

MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_14205*
value_dtype0	
m

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name16314*
value_dtype0	
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:*
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

: *
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
: *
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:@ *
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:@*
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:@*
dtype0

embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	N*%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	N*
dtype0
s
serving_default_examplesPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
с
StatefulPartitionedCallStatefulPartitionedCallserving_default_examples
hash_tableConst_5Const_4Const_3embedding/embeddingsdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/bias*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_20625
 
StatefulPartitionedCall_1StatefulPartitionedCall
hash_tableConstConst_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__initializer_21618
э
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__initializer_21633
:
NoOpNoOp^PartitionedCall^StatefulPartitionedCall_1
Ч
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
оK
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*K
valueKBK BK
в
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
		tft_layer

signatures*
* 

	keras_api* 
;
	keras_api
_lookup_layer
_adapt_function*
 
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*

	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses* 
І
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias*
І
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias*
І
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses

:kernel
;bias*
Д
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
$B _saved_model_loader_tracked_dict* 
5
1
*2
+3
24
35
:6
;7*
5
0
*1
+2
23
34
:5
;6*
* 
А
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Htrace_0
Itrace_1
Jtrace_2
Ktrace_3* 
6
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_3* 
/
P	capture_1
Q	capture_2
R	capture_3* 
а
Siter

Tbeta_1

Ubeta_2
	Vdecay
Wlearning_ratemБ*mВ+mГ2mД3mЕ:mЖ;mЗvИ*vЙ+vК2vЛ3vМ:vН;vО*

Xserving_default* 
* 
* 
7
Y	keras_api
Zlookup_table
[token_counts*

\trace_0* 

0*

0*
* 

]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

btrace_0* 

ctrace_0* 
hb
VARIABLE_VALUEembedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses* 

itrace_0* 

jtrace_0* 

*0
+1*

*0
+1*
* 

knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

ptrace_0* 

qtrace_0* 
_Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_12/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

20
31*

20
31*
* 

rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

wtrace_0* 

xtrace_0* 
_Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_13/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

:0
;1*

:0
;1*
* 

ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*

~trace_0* 

trace_0* 
_Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_14/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
y
	_imported
_wrapped_function
_structured_inputs
_structured_outputs
_output_to_inputs_map* 
* 
C
0
1
2
3
4
5
6
7
	8*

0
1*
* 
* 
/
P	capture_1
Q	capture_2
R	capture_3* 
/
P	capture_1
Q	capture_2
R	capture_3* 
/
P	capture_1
Q	capture_2
R	capture_3* 
/
P	capture_1
Q	capture_2
R	capture_3* 
/
P	capture_1
Q	capture_2
R	capture_3* 
/
P	capture_1
Q	capture_2
R	capture_3* 
/
P	capture_1
Q	capture_2
R	capture_3* 
/
P	capture_1
Q	capture_2
R	capture_3* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
/
P	capture_1
Q	capture_2
R	capture_3* 
* 
V
_initializer
_create_resource
_initialize
_destroy_resource* 

_create_resource
_initialize
_destroy_resourceJ
tableAlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table*

	capture_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ќ
created_variables
	resources
trackable_objects
initializers
assets

signatures
$_self_saveable_object_factories
transform_fn* 
* 
* 
* 
* 
<
	variables
 	keras_api

Ёtotal

Ђcount*
M
Ѓ	variables
Є	keras_api

Ѕtotal

Іcount
Ї
_fn_kwargs*
* 

Јtrace_0* 

Љtrace_0* 

Њtrace_0* 

Ћtrace_0* 

Ќtrace_0* 

­trace_0* 
* 
* 
* 
* 
* 
* 

Ўserving_default* 
* 

Ё0
Ђ1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Ѕ0
І1*

Ѓ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
"
Џ	capture_1
А	capture_2* 
* 
* 
* 
* 
* 
* 
* 

VARIABLE_VALUEAdam/embedding/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_12/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_12/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_13/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_13/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_14/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_14/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/embedding/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_12/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_12/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_13/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_13/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_14/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_14/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
С
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1total_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOpConst_6*-
Tin&
$2"		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_21793
я
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameembedding/embeddingsdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateMutableHashTabletotal_1count_1totalcountAdam/embedding/embeddings/mAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/mAdam/dense_14/kernel/mAdam/dense_14/bias/mAdam/embedding/embeddings/vAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/vAdam/dense_14/kernel/vAdam/dense_14/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_21905ив

О
B__inference_model_4_layer_call_and_return_conditional_losses_21409

inputsY
Utext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_table_handleZ
Vtext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_default_value	0
,text_vectorization_5_string_lookup_5_equal_y3
/text_vectorization_5_string_lookup_5_selectv2_t	3
 embedding_embedding_lookup_21380:	N9
'dense_12_matmul_readvariableop_resource:@6
(dense_12_biasadd_readvariableop_resource:@9
'dense_13_matmul_readvariableop_resource:@ 6
(dense_13_biasadd_readvariableop_resource: 9
'dense_14_matmul_readvariableop_resource: 6
(dense_14_biasadd_readvariableop_resource:
identityЂdense_12/BiasAdd/ReadVariableOpЂdense_12/MatMul/ReadVariableOpЂdense_13/BiasAdd/ReadVariableOpЂdense_13/MatMul/ReadVariableOpЂdense_14/BiasAdd/ReadVariableOpЂdense_14/MatMul/ReadVariableOpЂembedding/embedding_lookupЂHtext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2m
tf.reshape_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџz
tf.reshape_4/ReshapeReshapeinputs#tf.reshape_4/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџs
 text_vectorization_5/StringLowerStringLowertf.reshape_4/Reshape:output:0*#
_output_shapes
:џџџџџџџџџд
'text_vectorization_5/StaticRegexReplaceStaticRegexReplace)text_vectorization_5/StringLower:output:0*#
_output_shapes
:џџџџџџџџџ*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite g
&text_vectorization_5/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B р
.text_vectorization_5/StringSplit/StringSplitV2StringSplitV20text_vectorization_5/StaticRegexReplace:output:0/text_vectorization_5/StringSplit/Const:output:0*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:
4text_vectorization_5/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
6text_vectorization_5/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
6text_vectorization_5/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ў
.text_vectorization_5/StringSplit/strided_sliceStridedSlice8text_vectorization_5/StringSplit/StringSplitV2:indices:0=text_vectorization_5/StringSplit/strided_slice/stack:output:0?text_vectorization_5/StringSplit/strided_slice/stack_1:output:0?text_vectorization_5/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask
6text_vectorization_5/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8text_vectorization_5/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8text_vectorization_5/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0text_vectorization_5/StringSplit/strided_slice_1StridedSlice6text_vectorization_5/StringSplit/StringSplitV2:shape:0?text_vectorization_5/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_5/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_5/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskе
Wtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_5/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџЬ
Ytext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_5/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ь
atext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:Ћ
atext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: с
`text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: Ї
etext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ъ
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 
`text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ­
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: в
_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: Ѓ
atext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :п
_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: в
_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: г
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: з
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: І
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 М
itext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџэ
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape[text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџу
dtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountltext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ 
^text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
Ytext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:џџџџџџџџџЌ
btext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R  
^text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : г
Ytext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:џџџџџџџџџ
Htext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Utext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_table_handle7text_vectorization_5/StringSplit/StringSplitV2:values:0Vtext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:џџџџџџџџџШ
*text_vectorization_5/string_lookup_5/EqualEqual7text_vectorization_5/StringSplit/StringSplitV2:values:0,text_vectorization_5_string_lookup_5_equal_y*
T0*#
_output_shapes
:џџџџџџџџџ
-text_vectorization_5/string_lookup_5/SelectV2SelectV2.text_vectorization_5/string_lookup_5/Equal:z:0/text_vectorization_5_string_lookup_5_selectv2_tQtext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:џџџџџџџџџ
-text_vectorization_5/string_lookup_5/IdentityIdentity6text_vectorization_5/string_lookup_5/SelectV2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџs
1text_vectorization_5/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
)text_vectorization_5/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"џџџџџџџџd       
8text_vectorization_5/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_5/RaggedToTensor/Const:output:06text_vectorization_5/string_lookup_5/Identity:output:0:text_vectorization_5/RaggedToTensor/default_value:output:09text_vectorization_5/StringSplit/strided_slice_1:output:07text_vectorization_5/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:џџџџџџџџџd*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS
embedding/embedding_lookupResourceGather embedding_embedding_lookup_21380Atext_vectorization_5/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*3
_class)
'%loc:@embedding/embedding_lookup/21380*+
_output_shapes
:џџџџџџџџџd*
dtype0П
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/21380*+
_output_shapes
:џџџџџџџџџd
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџds
1global_average_pooling1d_4/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Х
global_average_pooling1d_4/MeanMean.embedding/embedding_lookup/Identity_1:output:0:global_average_pooling1d_4/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_12/MatMulMatMul(global_average_pooling1d_4/Mean:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@b
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ b
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_14/MatMulMatMuldense_13/Relu:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџh
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџi
IdentityIdentitydense_14/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџї
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp^embedding/embedding_lookupI^text_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2
Htext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2Htext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Я
:
__inference__creator_21610
identityЂ
hash_tablem

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name16314*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
Ю
[
:__inference_transform_features_layer_1_layer_call_fn_21584
inputs_text
identityХ
PartitionedCallPartitionedCallinputs_text*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_20747`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:T P
'
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameinputs/text
Р

(__inference_dense_12_layer_call_fn_21528

inputs
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_20859o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Р

(__inference_dense_13_layer_call_fn_21548

inputs
unknown:@ 
	unknown_0: 
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_20876o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ф

л
'__inference_model_4_layer_call_fn_21118
text_xf
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	N
	unknown_4:@
	unknown_5:@
	unknown_6:@ 
	unknown_7: 
	unknown_8: 
	unknown_9:
identityЂStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCalltext_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_4_layer_call_and_return_conditional_losses_21066o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	text_xf:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Й
T
:__inference_transform_features_layer_1_layer_call_fn_20750
text
identityО
PartitionedCallPartitionedCalltext*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_20747`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:M I
'
_output_shapes
:џџџџџџџџџ

_user_specified_nametext

R
__inference_pruned_20478

inputs	
inputs_1
identity	

identity_1Q
inputs_copyIdentityinputs*
T0	*'
_output_shapes
:џџџџџџџџџ\
IdentityIdentityinputs_copy:output:0*
T0	*'
_output_shapes
:џџџџџџџџџU
inputs_1_copyIdentityinputs_1*
T0*'
_output_shapes
:џџџџџџџџџ[
StringLowerStringLowerinputs_1_copy:output:0*'
_output_shapes
:џџџџџџџџџ^

Identity_1IdentityStringLower:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:- )
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ
Л	
й
__inference_restore_fn_21666
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identityЂ2MutableHashTable_table_restore/LookupTableImportV2
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :I
Const_1Const*
_output_shapes
: *
dtype0*
value	B :N
IdentityIdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
Н
o
U__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_20776
text
identity9
ShapeShapetext*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask;
Shape_1Shapetext*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:џџџџџџџџџ*
dtype0	*
shape:џџџџџџџџџ
PartitionedCallPartitionedCallPlaceholderWithDefault:output:0text*
Tin
2	*
Tout
2	*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_pruned_20478`
IdentityIdentityPartitionedCall:output:1*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:M I
'
_output_shapes
:џџџџџџџџџ

_user_specified_nametext


є
C__inference_dense_12_layer_call_and_return_conditional_losses_20859

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч
q
U__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_20747

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask=
Shape_1Shapeinputs*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:џџџџџџџџџ*
dtype0	*
shape:џџџџџџџџџ
PartitionedCallPartitionedCallPlaceholderWithDefault:output:0inputs*
Tin
2	*
Tout
2	*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_pruned_20478`
IdentityIdentityPartitionedCall:output:1*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
р
v
U__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_21605
inputs_text
identity@
ShapeShapeinputs_text*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskB
Shape_1Shapeinputs_text*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0	*'
_output_shapes
:џџџџџџџџџ
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:џџџџџџџџџ*
dtype0	*
shape:џџџџџџџџџ
PartitionedCallPartitionedCallPlaceholderWithDefault:output:0inputs_text*
Tin
2	*
Tout
2	*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_pruned_20478`
IdentityIdentityPartitionedCall:output:1*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:T P
'
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameinputs/text
ёt
М
B__inference_model_4_layer_call_and_return_conditional_losses_21266
text_xfY
Utext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_table_handleZ
Vtext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_default_value	0
,text_vectorization_5_string_lookup_5_equal_y3
/text_vectorization_5_string_lookup_5_selectv2_t	"
embedding_21246:	N 
dense_12_21250:@
dense_12_21252:@ 
dense_13_21255:@ 
dense_13_21257:  
dense_14_21260: 
dense_14_21262:
identityЂ dense_12/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallЂ dense_14/StatefulPartitionedCallЂ!embedding/StatefulPartitionedCallЂHtext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2m
tf.reshape_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
tf.reshape_4/ReshapeReshapetext_xf#tf.reshape_4/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџs
 text_vectorization_5/StringLowerStringLowertf.reshape_4/Reshape:output:0*#
_output_shapes
:џџџџџџџџџд
'text_vectorization_5/StaticRegexReplaceStaticRegexReplace)text_vectorization_5/StringLower:output:0*#
_output_shapes
:џџџџџџџџџ*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite g
&text_vectorization_5/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B р
.text_vectorization_5/StringSplit/StringSplitV2StringSplitV20text_vectorization_5/StaticRegexReplace:output:0/text_vectorization_5/StringSplit/Const:output:0*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:
4text_vectorization_5/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
6text_vectorization_5/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
6text_vectorization_5/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ў
.text_vectorization_5/StringSplit/strided_sliceStridedSlice8text_vectorization_5/StringSplit/StringSplitV2:indices:0=text_vectorization_5/StringSplit/strided_slice/stack:output:0?text_vectorization_5/StringSplit/strided_slice/stack_1:output:0?text_vectorization_5/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask
6text_vectorization_5/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8text_vectorization_5/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8text_vectorization_5/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0text_vectorization_5/StringSplit/strided_slice_1StridedSlice6text_vectorization_5/StringSplit/StringSplitV2:shape:0?text_vectorization_5/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_5/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_5/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskе
Wtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_5/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџЬ
Ytext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_5/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ь
atext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:Ћ
atext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: с
`text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: Ї
etext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ъ
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 
`text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ­
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: в
_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: Ѓ
atext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :п
_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: в
_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: г
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: з
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: І
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 М
itext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџэ
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape[text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџу
dtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountltext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ 
^text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
Ytext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:џџџџџџџџџЌ
btext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R  
^text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : г
Ytext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:џџџџџџџџџ
Htext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Utext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_table_handle7text_vectorization_5/StringSplit/StringSplitV2:values:0Vtext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:џџџџџџџџџШ
*text_vectorization_5/string_lookup_5/EqualEqual7text_vectorization_5/StringSplit/StringSplitV2:values:0,text_vectorization_5_string_lookup_5_equal_y*
T0*#
_output_shapes
:џџџџџџџџџ
-text_vectorization_5/string_lookup_5/SelectV2SelectV2.text_vectorization_5/string_lookup_5/Equal:z:0/text_vectorization_5_string_lookup_5_selectv2_tQtext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:џџџџџџџџџ
-text_vectorization_5/string_lookup_5/IdentityIdentity6text_vectorization_5/string_lookup_5/SelectV2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџs
1text_vectorization_5/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
)text_vectorization_5/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"џџџџџџџџd       
8text_vectorization_5/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_5/RaggedToTensor/Const:output:06text_vectorization_5/string_lookup_5/Identity:output:0:text_vectorization_5/RaggedToTensor/default_value:output:09text_vectorization_5/StringSplit/strided_slice_1:output:07text_vectorization_5/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:џџџџџџџџџd*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS
!embedding/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_5/RaggedToTensor/RaggedTensorToTensor:result:0embedding_21246*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_20843џ
*global_average_pooling1d_4/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_global_average_pooling1d_4_layer_call_and_return_conditional_losses_20719
 dense_12/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_4/PartitionedCall:output:0dense_12_21250dense_12_21252*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_20859
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_21255dense_13_21257*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_20876
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_21260dense_14_21262*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_20893x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall"^embedding/StatefulPartitionedCallI^text_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2
Htext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2Htext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	text_xf:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

О
B__inference_model_4_layer_call_and_return_conditional_losses_21492

inputsY
Utext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_table_handleZ
Vtext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_default_value	0
,text_vectorization_5_string_lookup_5_equal_y3
/text_vectorization_5_string_lookup_5_selectv2_t	3
 embedding_embedding_lookup_21463:	N9
'dense_12_matmul_readvariableop_resource:@6
(dense_12_biasadd_readvariableop_resource:@9
'dense_13_matmul_readvariableop_resource:@ 6
(dense_13_biasadd_readvariableop_resource: 9
'dense_14_matmul_readvariableop_resource: 6
(dense_14_biasadd_readvariableop_resource:
identityЂdense_12/BiasAdd/ReadVariableOpЂdense_12/MatMul/ReadVariableOpЂdense_13/BiasAdd/ReadVariableOpЂdense_13/MatMul/ReadVariableOpЂdense_14/BiasAdd/ReadVariableOpЂdense_14/MatMul/ReadVariableOpЂembedding/embedding_lookupЂHtext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2m
tf.reshape_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџz
tf.reshape_4/ReshapeReshapeinputs#tf.reshape_4/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџs
 text_vectorization_5/StringLowerStringLowertf.reshape_4/Reshape:output:0*#
_output_shapes
:џџџџџџџџџд
'text_vectorization_5/StaticRegexReplaceStaticRegexReplace)text_vectorization_5/StringLower:output:0*#
_output_shapes
:џџџџџџџџџ*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite g
&text_vectorization_5/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B р
.text_vectorization_5/StringSplit/StringSplitV2StringSplitV20text_vectorization_5/StaticRegexReplace:output:0/text_vectorization_5/StringSplit/Const:output:0*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:
4text_vectorization_5/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
6text_vectorization_5/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
6text_vectorization_5/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ў
.text_vectorization_5/StringSplit/strided_sliceStridedSlice8text_vectorization_5/StringSplit/StringSplitV2:indices:0=text_vectorization_5/StringSplit/strided_slice/stack:output:0?text_vectorization_5/StringSplit/strided_slice/stack_1:output:0?text_vectorization_5/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask
6text_vectorization_5/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8text_vectorization_5/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8text_vectorization_5/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0text_vectorization_5/StringSplit/strided_slice_1StridedSlice6text_vectorization_5/StringSplit/StringSplitV2:shape:0?text_vectorization_5/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_5/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_5/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskе
Wtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_5/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџЬ
Ytext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_5/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ь
atext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:Ћ
atext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: с
`text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: Ї
etext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ъ
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 
`text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ­
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: в
_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: Ѓ
atext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :п
_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: в
_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: г
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: з
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: І
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 М
itext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџэ
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape[text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџу
dtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountltext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ 
^text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
Ytext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:џџџџџџџџџЌ
btext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R  
^text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : г
Ytext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:џџџџџџџџџ
Htext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Utext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_table_handle7text_vectorization_5/StringSplit/StringSplitV2:values:0Vtext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:џџџџџџџџџШ
*text_vectorization_5/string_lookup_5/EqualEqual7text_vectorization_5/StringSplit/StringSplitV2:values:0,text_vectorization_5_string_lookup_5_equal_y*
T0*#
_output_shapes
:џџџџџџџџџ
-text_vectorization_5/string_lookup_5/SelectV2SelectV2.text_vectorization_5/string_lookup_5/Equal:z:0/text_vectorization_5_string_lookup_5_selectv2_tQtext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:џџџџџџџџџ
-text_vectorization_5/string_lookup_5/IdentityIdentity6text_vectorization_5/string_lookup_5/SelectV2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџs
1text_vectorization_5/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
)text_vectorization_5/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"џџџџџџџџd       
8text_vectorization_5/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_5/RaggedToTensor/Const:output:06text_vectorization_5/string_lookup_5/Identity:output:0:text_vectorization_5/RaggedToTensor/default_value:output:09text_vectorization_5/StringSplit/strided_slice_1:output:07text_vectorization_5/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:џџџџџџџџџd*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS
embedding/embedding_lookupResourceGather embedding_embedding_lookup_21463Atext_vectorization_5/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*3
_class)
'%loc:@embedding/embedding_lookup/21463*+
_output_shapes
:џџџџџџџџџd*
dtype0П
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/21463*+
_output_shapes
:џџџџџџџџџd
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџds
1global_average_pooling1d_4/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Х
global_average_pooling1d_4/MeanMean.embedding/embedding_lookup/Identity_1:output:0:global_average_pooling1d_4/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_12/MatMulMatMul(global_average_pooling1d_4/Mean:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@b
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ b
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_14/MatMulMatMuldense_13/Relu:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџh
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџi
IdentityIdentitydense_14/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџї
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp^embedding/embedding_lookupI^text_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2
Htext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2Htext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ф

л
'__inference_model_4_layer_call_fn_20925
text_xf
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	N
	unknown_4:@
	unknown_5:@
	unknown_6:@ 
	unknown_7: 
	unknown_8: 
	unknown_9:
identityЂStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCalltext_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_4_layer_call_and_return_conditional_losses_20900o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	text_xf:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


є
C__inference_dense_14_layer_call_and_return_conditional_losses_20893

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
§
V
:__inference_global_average_pooling1d_4_layer_call_fn_21513

inputs
identityЩ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_global_average_pooling1d_4_layer_call_and_return_conditional_losses_20719i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ТE
в
__inference__traced_save_21793
file_prefix3
/savev2_embedding_embeddings_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableop
savev2_const_6

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ѓ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*
valueB!B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЏ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B З
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!		
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ф
_input_shapesв
Я: :	N:@:@:@ : : :: : : : : ::: : : : :	N:@:@:@ : : ::	N:@:@:@ : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	N:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	N:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	N:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: :  

_output_shapes
::!

_output_shapes
: 


У
&__inference_restore_from_tensors_21864M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: @
<mutablehashtable_table_restore_lookuptableimportv2_restorev2B
>mutablehashtable_table_restore_lookuptableimportv2_restorev2_1	
identityЂ2MutableHashTable_table_restore/LookupTableImportV2о
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable<mutablehashtable_table_restore_lookuptableimportv2_restorev2>mutablehashtable_table_restore_lookuptableimportv2_restorev2_1*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*
_input_shapes

: ::2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:) %
#
_class
loc:@MutableHashTable:C?
#
_class
loc:@MutableHashTable

_output_shapes
::C?
#
_class
loc:@MutableHashTable

_output_shapes
:
П

и
#__inference_signature_wrapper_20625
examples
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	N
	unknown_4:@
	unknown_5:@
	unknown_6:@ 
	unknown_7: 
	unknown_8: 
	unknown_9:
identityЂStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallexamplesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_serve_tf_examples_fn_20596o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
examples:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
і
]
#__inference_signature_wrapper_20486

inputs	
inputs_1
identity	

identity_1
PartitionedCallPartitionedCallinputsinputs_1*
Tin
2	*
Tout
2	*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_pruned_20478`
IdentityIdentityPartitionedCall:output:0*
T0	*'
_output_shapes
:џџџџџџџџџb

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1
Р

(__inference_dense_14_layer_call_fn_21568

inputs
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_20893o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
џ}
Ў
!__inference__traced_restore_21905
file_prefix8
%assignvariableop_embedding_embeddings:	N4
"assignvariableop_1_dense_12_kernel:@.
 assignvariableop_2_dense_12_bias:@4
"assignvariableop_3_dense_13_kernel:@ .
 assignvariableop_4_dense_13_bias: 4
"assignvariableop_5_dense_14_kernel: .
 assignvariableop_6_dense_14_bias:&
assignvariableop_7_adam_iter:	 (
assignvariableop_8_adam_beta_1: (
assignvariableop_9_adam_beta_2: (
assignvariableop_10_adam_decay: 0
&assignvariableop_11_adam_learning_rate: 
mutablehashtable: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: #
assignvariableop_14_total: #
assignvariableop_15_count: B
/assignvariableop_16_adam_embedding_embeddings_m:	N<
*assignvariableop_17_adam_dense_12_kernel_m:@6
(assignvariableop_18_adam_dense_12_bias_m:@<
*assignvariableop_19_adam_dense_13_kernel_m:@ 6
(assignvariableop_20_adam_dense_13_bias_m: <
*assignvariableop_21_adam_dense_14_kernel_m: 6
(assignvariableop_22_adam_dense_14_bias_m:B
/assignvariableop_23_adam_embedding_embeddings_v:	N<
*assignvariableop_24_adam_dense_12_kernel_v:@6
(assignvariableop_25_adam_dense_12_bias_v:@<
*assignvariableop_26_adam_dense_13_kernel_v:@ 6
(assignvariableop_27_adam_dense_13_bias_v: <
*assignvariableop_28_adam_dense_14_kernel_v: 6
(assignvariableop_29_adam_dense_14_bias_v:
identity_31ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9ЂStatefulPartitionedCallі
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*
valueB!B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHВ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ц
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:::::::::::::::::::::::::::::::::*/
dtypes%
#2!		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_12_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense_12_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_13_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_13_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_14_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_14_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_iterIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_2Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_decayIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp&assignvariableop_11_adam_learning_rateIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0з
StatefulPartitionedCallStatefulPartitionedCallmutablehashtableRestoreV2:tensors:12RestoreV2:tensors:13"/device:CPU:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_restore_from_tensors_21864_
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_16AssignVariableOp/assignvariableop_16_adam_embedding_embeddings_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_12_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_12_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_13_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_13_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_14_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_14_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_23AssignVariableOp/assignvariableop_23_adam_embedding_embeddings_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_12_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_dense_12_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_13_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_dense_13_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_14_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense_14_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 §
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp^StatefulPartitionedCall"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: ъ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_922
StatefulPartitionedCallStatefulPartitionedCall:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

q
U__inference_global_average_pooling1d_4_layer_call_and_return_conditional_losses_20719

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
с

к
'__inference_model_4_layer_call_fn_21326

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	N
	unknown_4:@
	unknown_5:@
	unknown_6:@ 
	unknown_7: 
	unknown_8: 
	unknown_9:
identityЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_4_layer_call_and_return_conditional_losses_21066o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


є
C__inference_dense_14_layer_call_and_return_conditional_losses_21579

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ЄF

__inference_adapt_step_19692
iterator9
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	ЂIteratorGetNextЂ(None_lookup_table_find/LookupTableFindV2Ђ,None_lookup_table_insert/LookupTableInsertV2Љ
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:џџџџџџџџџ*"
output_shapes
:џџџџџџџџџ*
output_types
2]
StringLowerStringLowerIteratorGetNext:components:0*#
_output_shapes
:џџџџџџџџџЊ
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*#
_output_shapes
:џџџџџџџџџ*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B Ё
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace:output:0StringSplit/Const:output:0*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЋ
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџЂ
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: Т
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ђ
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : Ћ
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: з
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B : 
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 Ї
TStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџЎ
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0]StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountWStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : Ј
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : џ
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:џџџџџџџџџЃ
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
out_idx0	Ё
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:

_output_shapes
: 
Ф
Ё
D__inference_embedding_layer_call_and_return_conditional_losses_20843

inputs	)
embedding_lookup_20837:	N
identityЂembedding_lookupЗ
embedding_lookupResourceGatherembedding_lookup_20837inputs*
Tindices0	*)
_class
loc:@embedding_lookup/20837*+
_output_shapes
:џџџџџџџџџd*
dtype0Ё
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/20837*+
_output_shapes
:џџџџџџџџџd
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџdw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџdY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџd: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs

Е	
 __inference__wrapped_model_20709
text_xfa
]model_4_text_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_table_handleb
^model_4_text_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_default_value	8
4model_4_text_vectorization_5_string_lookup_5_equal_y;
7model_4_text_vectorization_5_string_lookup_5_selectv2_t	;
(model_4_embedding_embedding_lookup_20680:	NA
/model_4_dense_12_matmul_readvariableop_resource:@>
0model_4_dense_12_biasadd_readvariableop_resource:@A
/model_4_dense_13_matmul_readvariableop_resource:@ >
0model_4_dense_13_biasadd_readvariableop_resource: A
/model_4_dense_14_matmul_readvariableop_resource: >
0model_4_dense_14_biasadd_readvariableop_resource:
identityЂ'model_4/dense_12/BiasAdd/ReadVariableOpЂ&model_4/dense_12/MatMul/ReadVariableOpЂ'model_4/dense_13/BiasAdd/ReadVariableOpЂ&model_4/dense_13/MatMul/ReadVariableOpЂ'model_4/dense_14/BiasAdd/ReadVariableOpЂ&model_4/dense_14/MatMul/ReadVariableOpЂ"model_4/embedding/embedding_lookupЂPmodel_4/text_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2u
"model_4/tf.reshape_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
model_4/tf.reshape_4/ReshapeReshapetext_xf+model_4/tf.reshape_4/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
(model_4/text_vectorization_5/StringLowerStringLower%model_4/tf.reshape_4/Reshape:output:0*#
_output_shapes
:џџџџџџџџџф
/model_4/text_vectorization_5/StaticRegexReplaceStaticRegexReplace1model_4/text_vectorization_5/StringLower:output:0*#
_output_shapes
:џџџџџџџџџ*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite o
.model_4/text_vectorization_5/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ј
6model_4/text_vectorization_5/StringSplit/StringSplitV2StringSplitV28model_4/text_vectorization_5/StaticRegexReplace:output:07model_4/text_vectorization_5/StringSplit/Const:output:0*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:
<model_4/text_vectorization_5/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
>model_4/text_vectorization_5/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
>model_4/text_vectorization_5/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ж
6model_4/text_vectorization_5/StringSplit/strided_sliceStridedSlice@model_4/text_vectorization_5/StringSplit/StringSplitV2:indices:0Emodel_4/text_vectorization_5/StringSplit/strided_slice/stack:output:0Gmodel_4/text_vectorization_5/StringSplit/strided_slice/stack_1:output:0Gmodel_4/text_vectorization_5/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask
>model_4/text_vectorization_5/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
@model_4/text_vectorization_5/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
@model_4/text_vectorization_5/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
8model_4/text_vectorization_5/StringSplit/strided_slice_1StridedSlice>model_4/text_vectorization_5/StringSplit/StringSplitV2:shape:0Gmodel_4/text_vectorization_5/StringSplit/strided_slice_1/stack:output:0Imodel_4/text_vectorization_5/StringSplit/strided_slice_1/stack_1:output:0Imodel_4/text_vectorization_5/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskх
_model_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast?model_4/text_vectorization_5/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџм
amodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastAmodel_4/text_vectorization_5/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ќ
imodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapecmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:Г
imodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: љ
hmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdrmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0rmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: Џ
mmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 
kmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterqmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0vmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 
hmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastomodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: Е
kmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ъ
gmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxcmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0tmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: Ћ
imodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :ї
gmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2pmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0rmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ъ
gmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMullmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0kmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ы
kmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumemodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0kmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: я
kmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumemodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0omodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: Ў
kmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 Ф
qmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
kmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapecmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0zmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
lmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincounttmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0omodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0tmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџЈ
fmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : џ
amodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumsmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0omodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:џџџџџџџџџД
jmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R Ј
fmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ѓ
amodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2smodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0gmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0omodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:џџџџџџџџџЁ
Pmodel_4/text_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2LookupTableFindV2]model_4_text_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_table_handle?model_4/text_vectorization_5/StringSplit/StringSplitV2:values:0^model_4_text_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:џџџџџџџџџр
2model_4/text_vectorization_5/string_lookup_5/EqualEqual?model_4/text_vectorization_5/StringSplit/StringSplitV2:values:04model_4_text_vectorization_5_string_lookup_5_equal_y*
T0*#
_output_shapes
:џџџџџџџџџЛ
5model_4/text_vectorization_5/string_lookup_5/SelectV2SelectV26model_4/text_vectorization_5/string_lookup_5/Equal:z:07model_4_text_vectorization_5_string_lookup_5_selectv2_tYmodel_4/text_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:џџџџџџџџџЏ
5model_4/text_vectorization_5/string_lookup_5/IdentityIdentity>model_4/text_vectorization_5/string_lookup_5/SelectV2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ{
9model_4/text_vectorization_5/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
1model_4/text_vectorization_5/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"џџџџџџџџd       Ф
@model_4/text_vectorization_5/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor:model_4/text_vectorization_5/RaggedToTensor/Const:output:0>model_4/text_vectorization_5/string_lookup_5/Identity:output:0Bmodel_4/text_vectorization_5/RaggedToTensor/default_value:output:0Amodel_4/text_vectorization_5/StringSplit/strided_slice_1:output:0?model_4/text_vectorization_5/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:џџџџџџџџџd*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSА
"model_4/embedding/embedding_lookupResourceGather(model_4_embedding_embedding_lookup_20680Imodel_4/text_vectorization_5/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*;
_class1
/-loc:@model_4/embedding/embedding_lookup/20680*+
_output_shapes
:џџџџџџџџџd*
dtype0з
+model_4/embedding/embedding_lookup/IdentityIdentity+model_4/embedding/embedding_lookup:output:0*
T0*;
_class1
/-loc:@model_4/embedding/embedding_lookup/20680*+
_output_shapes
:џџџџџџџџџdЅ
-model_4/embedding/embedding_lookup/Identity_1Identity4model_4/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџd{
9model_4/global_average_pooling1d_4/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :н
'model_4/global_average_pooling1d_4/MeanMean6model_4/embedding/embedding_lookup/Identity_1:output:0Bmodel_4/global_average_pooling1d_4/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
&model_4/dense_12/MatMul/ReadVariableOpReadVariableOp/model_4_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Е
model_4/dense_12/MatMulMatMul0model_4/global_average_pooling1d_4/Mean:output:0.model_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
'model_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
model_4/dense_12/BiasAddBiasAdd!model_4/dense_12/MatMul:product:0/model_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
model_4/dense_12/ReluRelu!model_4/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
&model_4/dense_13/MatMul/ReadVariableOpReadVariableOp/model_4_dense_13_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0Ј
model_4/dense_13/MatMulMatMul#model_4/dense_12/Relu:activations:0.model_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
'model_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Љ
model_4/dense_13/BiasAddBiasAdd!model_4/dense_13/MatMul:product:0/model_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
model_4/dense_13/ReluRelu!model_4/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
&model_4/dense_14/MatMul/ReadVariableOpReadVariableOp/model_4_dense_14_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
model_4/dense_14/MatMulMatMul#model_4/dense_13/Relu:activations:0.model_4/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'model_4/dense_14/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
model_4/dense_14/BiasAddBiasAdd!model_4/dense_14/MatMul:product:0/model_4/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
model_4/dense_14/SoftmaxSoftmax!model_4/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџq
IdentityIdentity"model_4/dense_14/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЗ
NoOpNoOp(^model_4/dense_12/BiasAdd/ReadVariableOp'^model_4/dense_12/MatMul/ReadVariableOp(^model_4/dense_13/BiasAdd/ReadVariableOp'^model_4/dense_13/MatMul/ReadVariableOp(^model_4/dense_14/BiasAdd/ReadVariableOp'^model_4/dense_14/MatMul/ReadVariableOp#^model_4/embedding/embedding_lookupQ^model_4/text_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 2R
'model_4/dense_12/BiasAdd/ReadVariableOp'model_4/dense_12/BiasAdd/ReadVariableOp2P
&model_4/dense_12/MatMul/ReadVariableOp&model_4/dense_12/MatMul/ReadVariableOp2R
'model_4/dense_13/BiasAdd/ReadVariableOp'model_4/dense_13/BiasAdd/ReadVariableOp2P
&model_4/dense_13/MatMul/ReadVariableOp&model_4/dense_13/MatMul/ReadVariableOp2R
'model_4/dense_14/BiasAdd/ReadVariableOp'model_4/dense_14/BiasAdd/ReadVariableOp2P
&model_4/dense_14/MatMul/ReadVariableOp&model_4/dense_14/MatMul/ReadVariableOp2H
"model_4/embedding/embedding_lookup"model_4/embedding/embedding_lookup2Є
Pmodel_4/text_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2Pmodel_4/text_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	text_xf:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
юt
Л
B__inference_model_4_layer_call_and_return_conditional_losses_20900

inputsY
Utext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_table_handleZ
Vtext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_default_value	0
,text_vectorization_5_string_lookup_5_equal_y3
/text_vectorization_5_string_lookup_5_selectv2_t	"
embedding_20844:	N 
dense_12_20860:@
dense_12_20862:@ 
dense_13_20877:@ 
dense_13_20879:  
dense_14_20894: 
dense_14_20896:
identityЂ dense_12/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallЂ dense_14/StatefulPartitionedCallЂ!embedding/StatefulPartitionedCallЂHtext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2m
tf.reshape_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџz
tf.reshape_4/ReshapeReshapeinputs#tf.reshape_4/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџs
 text_vectorization_5/StringLowerStringLowertf.reshape_4/Reshape:output:0*#
_output_shapes
:џџџџџџџџџд
'text_vectorization_5/StaticRegexReplaceStaticRegexReplace)text_vectorization_5/StringLower:output:0*#
_output_shapes
:џџџџџџџџџ*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite g
&text_vectorization_5/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B р
.text_vectorization_5/StringSplit/StringSplitV2StringSplitV20text_vectorization_5/StaticRegexReplace:output:0/text_vectorization_5/StringSplit/Const:output:0*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:
4text_vectorization_5/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
6text_vectorization_5/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
6text_vectorization_5/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ў
.text_vectorization_5/StringSplit/strided_sliceStridedSlice8text_vectorization_5/StringSplit/StringSplitV2:indices:0=text_vectorization_5/StringSplit/strided_slice/stack:output:0?text_vectorization_5/StringSplit/strided_slice/stack_1:output:0?text_vectorization_5/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask
6text_vectorization_5/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8text_vectorization_5/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8text_vectorization_5/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0text_vectorization_5/StringSplit/strided_slice_1StridedSlice6text_vectorization_5/StringSplit/StringSplitV2:shape:0?text_vectorization_5/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_5/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_5/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskе
Wtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_5/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџЬ
Ytext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_5/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ь
atext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:Ћ
atext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: с
`text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: Ї
etext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ъ
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 
`text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ­
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: в
_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: Ѓ
atext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :п
_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: в
_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: г
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: з
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: І
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 М
itext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџэ
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape[text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџу
dtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountltext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ 
^text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
Ytext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:џџџџџџџџџЌ
btext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R  
^text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : г
Ytext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:џџџџџџџџџ
Htext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Utext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_table_handle7text_vectorization_5/StringSplit/StringSplitV2:values:0Vtext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:џџџџџџџџџШ
*text_vectorization_5/string_lookup_5/EqualEqual7text_vectorization_5/StringSplit/StringSplitV2:values:0,text_vectorization_5_string_lookup_5_equal_y*
T0*#
_output_shapes
:џџџџџџџџџ
-text_vectorization_5/string_lookup_5/SelectV2SelectV2.text_vectorization_5/string_lookup_5/Equal:z:0/text_vectorization_5_string_lookup_5_selectv2_tQtext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:џџџџџџџџџ
-text_vectorization_5/string_lookup_5/IdentityIdentity6text_vectorization_5/string_lookup_5/SelectV2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџs
1text_vectorization_5/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
)text_vectorization_5/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"џџџџџџџџd       
8text_vectorization_5/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_5/RaggedToTensor/Const:output:06text_vectorization_5/string_lookup_5/Identity:output:0:text_vectorization_5/RaggedToTensor/default_value:output:09text_vectorization_5/StringSplit/strided_slice_1:output:07text_vectorization_5/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:џџџџџџџџџd*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS
!embedding/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_5/RaggedToTensor/RaggedTensorToTensor:result:0embedding_20844*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_20843џ
*global_average_pooling1d_4/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_global_average_pooling1d_4_layer_call_and_return_conditional_losses_20719
 dense_12/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_4/PartitionedCall:output:0dense_12_20860dense_12_20862*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_20859
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_20877dense_13_20879*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_20876
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_20894dense_14_20896*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_20893x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall"^embedding/StatefulPartitionedCallI^text_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2
Htext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2Htext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
юt
Л
B__inference_model_4_layer_call_and_return_conditional_losses_21066

inputsY
Utext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_table_handleZ
Vtext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_default_value	0
,text_vectorization_5_string_lookup_5_equal_y3
/text_vectorization_5_string_lookup_5_selectv2_t	"
embedding_21046:	N 
dense_12_21050:@
dense_12_21052:@ 
dense_13_21055:@ 
dense_13_21057:  
dense_14_21060: 
dense_14_21062:
identityЂ dense_12/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallЂ dense_14/StatefulPartitionedCallЂ!embedding/StatefulPartitionedCallЂHtext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2m
tf.reshape_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџz
tf.reshape_4/ReshapeReshapeinputs#tf.reshape_4/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџs
 text_vectorization_5/StringLowerStringLowertf.reshape_4/Reshape:output:0*#
_output_shapes
:џџџџџџџџџд
'text_vectorization_5/StaticRegexReplaceStaticRegexReplace)text_vectorization_5/StringLower:output:0*#
_output_shapes
:џџџџџџџџџ*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite g
&text_vectorization_5/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B р
.text_vectorization_5/StringSplit/StringSplitV2StringSplitV20text_vectorization_5/StaticRegexReplace:output:0/text_vectorization_5/StringSplit/Const:output:0*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:
4text_vectorization_5/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
6text_vectorization_5/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
6text_vectorization_5/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ў
.text_vectorization_5/StringSplit/strided_sliceStridedSlice8text_vectorization_5/StringSplit/StringSplitV2:indices:0=text_vectorization_5/StringSplit/strided_slice/stack:output:0?text_vectorization_5/StringSplit/strided_slice/stack_1:output:0?text_vectorization_5/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask
6text_vectorization_5/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8text_vectorization_5/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8text_vectorization_5/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0text_vectorization_5/StringSplit/strided_slice_1StridedSlice6text_vectorization_5/StringSplit/StringSplitV2:shape:0?text_vectorization_5/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_5/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_5/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskе
Wtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_5/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџЬ
Ytext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_5/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ь
atext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:Ћ
atext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: с
`text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: Ї
etext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ъ
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 
`text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ­
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: в
_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: Ѓ
atext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :п
_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: в
_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: г
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: з
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: І
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 М
itext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџэ
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape[text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџу
dtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountltext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ 
^text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
Ytext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:џџџџџџџџџЌ
btext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R  
^text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : г
Ytext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:џџџџџџџџџ
Htext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Utext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_table_handle7text_vectorization_5/StringSplit/StringSplitV2:values:0Vtext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:џџџџџџџџџШ
*text_vectorization_5/string_lookup_5/EqualEqual7text_vectorization_5/StringSplit/StringSplitV2:values:0,text_vectorization_5_string_lookup_5_equal_y*
T0*#
_output_shapes
:џџџџџџџџџ
-text_vectorization_5/string_lookup_5/SelectV2SelectV2.text_vectorization_5/string_lookup_5/Equal:z:0/text_vectorization_5_string_lookup_5_selectv2_tQtext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:џџџџџџџџџ
-text_vectorization_5/string_lookup_5/IdentityIdentity6text_vectorization_5/string_lookup_5/SelectV2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџs
1text_vectorization_5/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
)text_vectorization_5/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"џџџџџџџџd       
8text_vectorization_5/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_5/RaggedToTensor/Const:output:06text_vectorization_5/string_lookup_5/Identity:output:0:text_vectorization_5/RaggedToTensor/default_value:output:09text_vectorization_5/StringSplit/strided_slice_1:output:07text_vectorization_5/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:џџџџџџџџџd*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS
!embedding/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_5/RaggedToTensor/RaggedTensorToTensor:result:0embedding_21046*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_20843џ
*global_average_pooling1d_4/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_global_average_pooling1d_4_layer_call_and_return_conditional_losses_20719
 dense_12/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_4/PartitionedCall:output:0dense_12_21050dense_12_21052*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_20859
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_21055dense_13_21057*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_20876
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_21060dense_14_21062*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_20893x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall"^embedding/StatefulPartitionedCallI^text_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2
Htext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2Htext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ХЊ
М	
&__inference_serve_tf_examples_fn_20596
examplesa
]model_4_text_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_table_handleb
^model_4_text_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_default_value	8
4model_4_text_vectorization_5_string_lookup_5_equal_y;
7model_4_text_vectorization_5_string_lookup_5_selectv2_t	;
(model_4_embedding_embedding_lookup_20567:	NA
/model_4_dense_12_matmul_readvariableop_resource:@>
0model_4_dense_12_biasadd_readvariableop_resource:@A
/model_4_dense_13_matmul_readvariableop_resource:@ >
0model_4_dense_13_biasadd_readvariableop_resource: A
/model_4_dense_14_matmul_readvariableop_resource: >
0model_4_dense_14_biasadd_readvariableop_resource:
identityЂ'model_4/dense_12/BiasAdd/ReadVariableOpЂ&model_4/dense_12/MatMul/ReadVariableOpЂ'model_4/dense_13/BiasAdd/ReadVariableOpЂ&model_4/dense_13/MatMul/ReadVariableOpЂ'model_4/dense_14/BiasAdd/ReadVariableOpЂ&model_4/dense_14/MatMul/ReadVariableOpЂ"model_4/embedding/embedding_lookupЂPmodel_4/text_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2U
ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB j
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB s
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*
valueBBtextj
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB У
ParseExample/ParseExampleV2ParseExampleV2examples*ParseExample/ParseExampleV2/names:output:00ParseExample/ParseExampleV2/sparse_keys:output:0/ParseExample/ParseExampleV2/dense_keys:output:00ParseExample/ParseExampleV2/ragged_keys:output:0ParseExample/Const:output:0*
Tdense
2*'
_output_shapes
:џџџџџџџџџ*
dense_shapes
:*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 z
 transform_features_layer_1/ShapeShape*ParseExample/ParseExampleV2:dense_values:0*
T0*
_output_shapes
:x
.transform_features_layer_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0transform_features_layer_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0transform_features_layer_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
(transform_features_layer_1/strided_sliceStridedSlice)transform_features_layer_1/Shape:output:07transform_features_layer_1/strided_slice/stack:output:09transform_features_layer_1/strided_slice/stack_1:output:09transform_features_layer_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
"transform_features_layer_1/Shape_1Shape*ParseExample/ParseExampleV2:dense_values:0*
T0*
_output_shapes
:z
0transform_features_layer_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2transform_features_layer_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2transform_features_layer_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
*transform_features_layer_1/strided_slice_1StridedSlice+transform_features_layer_1/Shape_1:output:09transform_features_layer_1/strided_slice_1/stack:output:0;transform_features_layer_1/strided_slice_1/stack_1:output:0;transform_features_layer_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)transform_features_layer_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ц
'transform_features_layer_1/zeros/packedPack3transform_features_layer_1/strided_slice_1:output:02transform_features_layer_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:h
&transform_features_layer_1/zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Н
 transform_features_layer_1/zerosFill0transform_features_layer_1/zeros/packed:output:0/transform_features_layer_1/zeros/Const:output:0*
T0	*'
_output_shapes
:џџџџџџџџџЪ
1transform_features_layer_1/PlaceholderWithDefaultPlaceholderWithDefault)transform_features_layer_1/zeros:output:0*'
_output_shapes
:џџџџџџџџџ*
dtype0	*
shape:џџџџџџџџџє
*transform_features_layer_1/PartitionedCallPartitionedCall:transform_features_layer_1/PlaceholderWithDefault:output:0*ParseExample/ParseExampleV2:dense_values:0*
Tin
2	*
Tout
2	*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_pruned_20478u
"model_4/tf.reshape_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџЗ
model_4/tf.reshape_4/ReshapeReshape3transform_features_layer_1/PartitionedCall:output:1+model_4/tf.reshape_4/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
(model_4/text_vectorization_5/StringLowerStringLower%model_4/tf.reshape_4/Reshape:output:0*#
_output_shapes
:џџџџџџџџџф
/model_4/text_vectorization_5/StaticRegexReplaceStaticRegexReplace1model_4/text_vectorization_5/StringLower:output:0*#
_output_shapes
:џџџџџџџџџ*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite o
.model_4/text_vectorization_5/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ј
6model_4/text_vectorization_5/StringSplit/StringSplitV2StringSplitV28model_4/text_vectorization_5/StaticRegexReplace:output:07model_4/text_vectorization_5/StringSplit/Const:output:0*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:
<model_4/text_vectorization_5/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
>model_4/text_vectorization_5/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
>model_4/text_vectorization_5/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ж
6model_4/text_vectorization_5/StringSplit/strided_sliceStridedSlice@model_4/text_vectorization_5/StringSplit/StringSplitV2:indices:0Emodel_4/text_vectorization_5/StringSplit/strided_slice/stack:output:0Gmodel_4/text_vectorization_5/StringSplit/strided_slice/stack_1:output:0Gmodel_4/text_vectorization_5/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask
>model_4/text_vectorization_5/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
@model_4/text_vectorization_5/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
@model_4/text_vectorization_5/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
8model_4/text_vectorization_5/StringSplit/strided_slice_1StridedSlice>model_4/text_vectorization_5/StringSplit/StringSplitV2:shape:0Gmodel_4/text_vectorization_5/StringSplit/strided_slice_1/stack:output:0Imodel_4/text_vectorization_5/StringSplit/strided_slice_1/stack_1:output:0Imodel_4/text_vectorization_5/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskх
_model_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast?model_4/text_vectorization_5/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџм
amodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastAmodel_4/text_vectorization_5/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ќ
imodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapecmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:Г
imodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: љ
hmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdrmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0rmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: Џ
mmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 
kmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterqmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0vmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 
hmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastomodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: Е
kmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ъ
gmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxcmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0tmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: Ћ
imodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :ї
gmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2pmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0rmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ъ
gmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMullmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0kmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ы
kmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumemodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0kmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: я
kmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumemodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0omodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: Ў
kmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 Ф
qmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
kmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapecmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0zmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
lmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincounttmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0omodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0tmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџЈ
fmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : џ
amodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumsmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0omodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:џџџџџџџџџД
jmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R Ј
fmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ѓ
amodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2smodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0gmodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0omodel_4/text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:џџџџџџџџџЁ
Pmodel_4/text_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2LookupTableFindV2]model_4_text_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_table_handle?model_4/text_vectorization_5/StringSplit/StringSplitV2:values:0^model_4_text_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:џџџџџџџџџр
2model_4/text_vectorization_5/string_lookup_5/EqualEqual?model_4/text_vectorization_5/StringSplit/StringSplitV2:values:04model_4_text_vectorization_5_string_lookup_5_equal_y*
T0*#
_output_shapes
:џџџџџџџџџЛ
5model_4/text_vectorization_5/string_lookup_5/SelectV2SelectV26model_4/text_vectorization_5/string_lookup_5/Equal:z:07model_4_text_vectorization_5_string_lookup_5_selectv2_tYmodel_4/text_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:џџџџџџџџџЏ
5model_4/text_vectorization_5/string_lookup_5/IdentityIdentity>model_4/text_vectorization_5/string_lookup_5/SelectV2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ{
9model_4/text_vectorization_5/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
1model_4/text_vectorization_5/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"џџџџџџџџd       Ф
@model_4/text_vectorization_5/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor:model_4/text_vectorization_5/RaggedToTensor/Const:output:0>model_4/text_vectorization_5/string_lookup_5/Identity:output:0Bmodel_4/text_vectorization_5/RaggedToTensor/default_value:output:0Amodel_4/text_vectorization_5/StringSplit/strided_slice_1:output:0?model_4/text_vectorization_5/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:џџџџџџџџџd*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSА
"model_4/embedding/embedding_lookupResourceGather(model_4_embedding_embedding_lookup_20567Imodel_4/text_vectorization_5/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*;
_class1
/-loc:@model_4/embedding/embedding_lookup/20567*+
_output_shapes
:џџџџџџџџџd*
dtype0з
+model_4/embedding/embedding_lookup/IdentityIdentity+model_4/embedding/embedding_lookup:output:0*
T0*;
_class1
/-loc:@model_4/embedding/embedding_lookup/20567*+
_output_shapes
:џџџџџџџџџdЅ
-model_4/embedding/embedding_lookup/Identity_1Identity4model_4/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџd{
9model_4/global_average_pooling1d_4/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :н
'model_4/global_average_pooling1d_4/MeanMean6model_4/embedding/embedding_lookup/Identity_1:output:0Bmodel_4/global_average_pooling1d_4/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
&model_4/dense_12/MatMul/ReadVariableOpReadVariableOp/model_4_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Е
model_4/dense_12/MatMulMatMul0model_4/global_average_pooling1d_4/Mean:output:0.model_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
'model_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
model_4/dense_12/BiasAddBiasAdd!model_4/dense_12/MatMul:product:0/model_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
model_4/dense_12/ReluRelu!model_4/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
&model_4/dense_13/MatMul/ReadVariableOpReadVariableOp/model_4_dense_13_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0Ј
model_4/dense_13/MatMulMatMul#model_4/dense_12/Relu:activations:0.model_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
'model_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Љ
model_4/dense_13/BiasAddBiasAdd!model_4/dense_13/MatMul:product:0/model_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
model_4/dense_13/ReluRelu!model_4/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
&model_4/dense_14/MatMul/ReadVariableOpReadVariableOp/model_4_dense_14_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
model_4/dense_14/MatMulMatMul#model_4/dense_13/Relu:activations:0.model_4/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'model_4/dense_14/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
model_4/dense_14/BiasAddBiasAdd!model_4/dense_14/MatMul:product:0/model_4/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
model_4/dense_14/SoftmaxSoftmax!model_4/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџq
IdentityIdentity"model_4/dense_14/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЗ
NoOpNoOp(^model_4/dense_12/BiasAdd/ReadVariableOp'^model_4/dense_12/MatMul/ReadVariableOp(^model_4/dense_13/BiasAdd/ReadVariableOp'^model_4/dense_13/MatMul/ReadVariableOp(^model_4/dense_14/BiasAdd/ReadVariableOp'^model_4/dense_14/MatMul/ReadVariableOp#^model_4/embedding/embedding_lookupQ^model_4/text_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : : : : : : : : 2R
'model_4/dense_12/BiasAdd/ReadVariableOp'model_4/dense_12/BiasAdd/ReadVariableOp2P
&model_4/dense_12/MatMul/ReadVariableOp&model_4/dense_12/MatMul/ReadVariableOp2R
'model_4/dense_13/BiasAdd/ReadVariableOp'model_4/dense_13/BiasAdd/ReadVariableOp2P
&model_4/dense_13/MatMul/ReadVariableOp&model_4/dense_13/MatMul/ReadVariableOp2R
'model_4/dense_14/BiasAdd/ReadVariableOp'model_4/dense_14/BiasAdd/ReadVariableOp2P
&model_4/dense_14/MatMul/ReadVariableOp&model_4/dense_14/MatMul/ReadVariableOp2H
"model_4/embedding/embedding_lookup"model_4/embedding/embedding_lookup2Є
Pmodel_4/text_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2Pmodel_4/text_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2:M I
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
examples:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

ћ
__inference__initializer_216188
4key_value_init16313_lookuptableimportv2_table_handle0
,key_value_init16313_lookuptableimportv2_keys2
.key_value_init16313_lookuptableimportv2_values	
identityЂ'key_value_init16313/LookupTableImportV2џ
'key_value_init16313/LookupTableImportV2LookupTableImportV24key_value_init16313_lookuptableimportv2_table_handle,key_value_init16313_lookuptableimportv2_keys.key_value_init16313_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^key_value_init16313/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :Н	:Н	2R
'key_value_init16313/LookupTableImportV2'key_value_init16313/LookupTableImportV2:!

_output_shapes	
:Н	:!

_output_shapes	
:Н	

F
__inference__creator_21628
identity: ЂMutableHashTable
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_14205*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
ёt
М
B__inference_model_4_layer_call_and_return_conditional_losses_21192
text_xfY
Utext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_table_handleZ
Vtext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_default_value	0
,text_vectorization_5_string_lookup_5_equal_y3
/text_vectorization_5_string_lookup_5_selectv2_t	"
embedding_21172:	N 
dense_12_21176:@
dense_12_21178:@ 
dense_13_21181:@ 
dense_13_21183:  
dense_14_21186: 
dense_14_21188:
identityЂ dense_12/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallЂ dense_14/StatefulPartitionedCallЂ!embedding/StatefulPartitionedCallЂHtext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2m
tf.reshape_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
tf.reshape_4/ReshapeReshapetext_xf#tf.reshape_4/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџs
 text_vectorization_5/StringLowerStringLowertf.reshape_4/Reshape:output:0*#
_output_shapes
:џџџџџџџџџд
'text_vectorization_5/StaticRegexReplaceStaticRegexReplace)text_vectorization_5/StringLower:output:0*#
_output_shapes
:џџџџџџџџџ*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite g
&text_vectorization_5/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B р
.text_vectorization_5/StringSplit/StringSplitV2StringSplitV20text_vectorization_5/StaticRegexReplace:output:0/text_vectorization_5/StringSplit/Const:output:0*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:
4text_vectorization_5/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
6text_vectorization_5/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
6text_vectorization_5/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ў
.text_vectorization_5/StringSplit/strided_sliceStridedSlice8text_vectorization_5/StringSplit/StringSplitV2:indices:0=text_vectorization_5/StringSplit/strided_slice/stack:output:0?text_vectorization_5/StringSplit/strided_slice/stack_1:output:0?text_vectorization_5/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask
6text_vectorization_5/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8text_vectorization_5/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8text_vectorization_5/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0text_vectorization_5/StringSplit/strided_slice_1StridedSlice6text_vectorization_5/StringSplit/StringSplitV2:shape:0?text_vectorization_5/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_5/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_5/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskе
Wtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_5/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџЬ
Ytext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_5/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ь
atext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:Ћ
atext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: с
`text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: Ї
etext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ъ
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 
`text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ­
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: в
_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: Ѓ
atext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :п
_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: в
_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: г
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: з
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: І
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 М
itext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџэ
ctext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape[text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџу
dtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountltext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ 
^text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
Ytext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:џџџџџџџџџЌ
btext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R  
^text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : г
Ytext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_5/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:џџџџџџџџџ
Htext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Utext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_table_handle7text_vectorization_5/StringSplit/StringSplitV2:values:0Vtext_vectorization_5_string_lookup_5_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:џџџџџџџџџШ
*text_vectorization_5/string_lookup_5/EqualEqual7text_vectorization_5/StringSplit/StringSplitV2:values:0,text_vectorization_5_string_lookup_5_equal_y*
T0*#
_output_shapes
:џџџџџџџџџ
-text_vectorization_5/string_lookup_5/SelectV2SelectV2.text_vectorization_5/string_lookup_5/Equal:z:0/text_vectorization_5_string_lookup_5_selectv2_tQtext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:џџџџџџџџџ
-text_vectorization_5/string_lookup_5/IdentityIdentity6text_vectorization_5/string_lookup_5/SelectV2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџs
1text_vectorization_5/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
)text_vectorization_5/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"џџџџџџџџd       
8text_vectorization_5/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_5/RaggedToTensor/Const:output:06text_vectorization_5/string_lookup_5/Identity:output:0:text_vectorization_5/RaggedToTensor/default_value:output:09text_vectorization_5/StringSplit/strided_slice_1:output:07text_vectorization_5/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:џџџџџџџџџd*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS
!embedding/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_5/RaggedToTensor/RaggedTensorToTensor:result:0embedding_21172*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_20843џ
*global_average_pooling1d_4/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_global_average_pooling1d_4_layer_call_and_return_conditional_losses_20719
 dense_12/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_4/PartitionedCall:output:0dense_12_21176dense_12_21178*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_20859
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_21181dense_13_21183*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_20876
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_21186dense_14_21188*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_20893x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall"^embedding/StatefulPartitionedCallI^text_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2
Htext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2Htext_vectorization_5/string_lookup_5/hash_table_Lookup/LookupTableFindV2:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	text_xf:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ђ
~
)__inference_embedding_layer_call_fn_21499

inputs	
unknown:	N
identityЂStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_20843s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџd: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
с

к
'__inference_model_4_layer_call_fn_21299

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	N
	unknown_4:@
	unknown_5:@
	unknown_6:@ 
	unknown_7: 
	unknown_8: 
	unknown_9:
identityЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_4_layer_call_and_return_conditional_losses_20900o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

,
__inference__destroyer_21623
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 

.
__inference__initializer_21633
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 


є
C__inference_dense_13_layer_call_and_return_conditional_losses_21559

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


є
C__inference_dense_12_layer_call_and_return_conditional_losses_21539

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф
Ё
D__inference_embedding_layer_call_and_return_conditional_losses_21508

inputs	)
embedding_lookup_21502:	N
identityЂembedding_lookupЗ
embedding_lookupResourceGatherembedding_lookup_21502inputs*
Tindices0	*)
_class
loc:@embedding_lookup/21502*+
_output_shapes
:џџџџџџџџџd*
dtype0Ё
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/21502*+
_output_shapes
:џџџџџџџџџd
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџdw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџdY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџd: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs

,
__inference__destroyer_21638
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Й
Є
__inference_save_fn_21657
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	Ђ?MutableHashTable_lookup_table_export_values/LookupTableExportV2
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::K
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: O
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: 

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: 

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key

q
U__inference_global_average_pooling1d_4_layer_call_and_return_conditional_losses_21519

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


є
C__inference_dense_13_layer_call_and_return_conditional_losses_20876

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs"Е	L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Љ
serving_default
9
examples-
serving_default_examples:0џџџџџџџџџ<
output_00
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:х
щ
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
		tft_layer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
(
	keras_api"
_tf_keras_layer
P
	keras_api
_lookup_layer
_adapt_function"
_tf_keras_layer
Е
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
Ѕ
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias"
_tf_keras_layer
Л
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias"
_tf_keras_layer
Л
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses

:kernel
;bias"
_tf_keras_layer
Ы
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
$B _saved_model_loader_tracked_dict"
_tf_keras_model
Q
1
*2
+3
24
35
:6
;7"
trackable_list_wrapper
Q
0
*1
+2
23
34
:5
;6"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
б
Htrace_0
Itrace_1
Jtrace_2
Ktrace_32ц
'__inference_model_4_layer_call_fn_20925
'__inference_model_4_layer_call_fn_21299
'__inference_model_4_layer_call_fn_21326
'__inference_model_4_layer_call_fn_21118П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zHtrace_0zItrace_1zJtrace_2zKtrace_3
Н
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_32в
B__inference_model_4_layer_call_and_return_conditional_losses_21409
B__inference_model_4_layer_call_and_return_conditional_losses_21492
B__inference_model_4_layer_call_and_return_conditional_losses_21192
B__inference_model_4_layer_call_and_return_conditional_losses_21266П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zLtrace_0zMtrace_1zNtrace_2zOtrace_3
Ѕ
P	capture_1
Q	capture_2
R	capture_3BШ
 __inference__wrapped_model_20709text_xf"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zP	capture_1zQ	capture_2zR	capture_3
п
Siter

Tbeta_1

Ubeta_2
	Vdecay
Wlearning_ratemБ*mВ+mГ2mД3mЕ:mЖ;mЗvИ*vЙ+vК2vЛ3vМ:vН;vО"
	optimizer
,
Xserving_default"
signature_map
"
_generic_user_object
"
_generic_user_object
L
Y	keras_api
Zlookup_table
[token_counts"
_tf_keras_layer
и
\trace_02Л
__inference_adapt_step_19692
В
FullArgSpec
args

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z\trace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
э
btrace_02а
)__inference_embedding_layer_call_fn_21499Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zbtrace_0

ctrace_02ы
D__inference_embedding_layer_call_and_return_conditional_losses_21508Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zctrace_0
':%	N2embedding/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object

itrace_02ю
:__inference_global_average_pooling1d_4_layer_call_fn_21513Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zitrace_0
І
jtrace_02
U__inference_global_average_pooling1d_4_layer_call_and_return_conditional_losses_21519Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zjtrace_0
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
ь
ptrace_02Я
(__inference_dense_12_layer_call_fn_21528Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zptrace_0

qtrace_02ъ
C__inference_dense_12_layer_call_and_return_conditional_losses_21539Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zqtrace_0
!:@2dense_12/kernel
:@2dense_12/bias
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
­
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
ь
wtrace_02Я
(__inference_dense_13_layer_call_fn_21548Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zwtrace_0

xtrace_02ъ
C__inference_dense_13_layer_call_and_return_conditional_losses_21559Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zxtrace_0
!:@ 2dense_13/kernel
: 2dense_13/bias
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
ь
~trace_02Я
(__inference_dense_14_layer_call_fn_21568Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z~trace_0

trace_02ъ
C__inference_dense_14_layer_call_and_return_conditional_losses_21579Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
!: 2dense_14/kernel
:2dense_14/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
и
trace_0
trace_12
:__inference_transform_features_layer_1_layer_call_fn_20750
:__inference_transform_features_layer_1_layer_call_fn_21584Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12г
U__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_21605
U__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_20776Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

	_imported
_wrapped_function
_structured_inputs
_structured_outputs
_output_to_inputs_map"
trackable_dict_wrapper
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
г
P	capture_1
Q	capture_2
R	capture_3Bі
'__inference_model_4_layer_call_fn_20925text_xf"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zP	capture_1zQ	capture_2zR	capture_3
в
P	capture_1
Q	capture_2
R	capture_3Bѕ
'__inference_model_4_layer_call_fn_21299inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zP	capture_1zQ	capture_2zR	capture_3
в
P	capture_1
Q	capture_2
R	capture_3Bѕ
'__inference_model_4_layer_call_fn_21326inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zP	capture_1zQ	capture_2zR	capture_3
г
P	capture_1
Q	capture_2
R	capture_3Bі
'__inference_model_4_layer_call_fn_21118text_xf"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zP	capture_1zQ	capture_2zR	capture_3
э
P	capture_1
Q	capture_2
R	capture_3B
B__inference_model_4_layer_call_and_return_conditional_losses_21409inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zP	capture_1zQ	capture_2zR	capture_3
э
P	capture_1
Q	capture_2
R	capture_3B
B__inference_model_4_layer_call_and_return_conditional_losses_21492inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zP	capture_1zQ	capture_2zR	capture_3
ю
P	capture_1
Q	capture_2
R	capture_3B
B__inference_model_4_layer_call_and_return_conditional_losses_21192text_xf"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zP	capture_1zQ	capture_2zR	capture_3
ю
P	capture_1
Q	capture_2
R	capture_3B
B__inference_model_4_layer_call_and_return_conditional_losses_21266text_xf"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zP	capture_1zQ	capture_2zR	capture_3
!J	
Const_5jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
Ѕ
P	capture_1
Q	capture_2
R	capture_3BШ
#__inference_signature_wrapper_20625examples"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zP	capture_1zQ	capture_2zR	capture_3
"
_generic_user_object
j
_initializer
_create_resource
_initialize
_destroy_resourceR jtf.StaticHashTable
T
_create_resource
_initialize
_destroy_resourceR Z
tableПР
ъ
	capture_1BЧ
__inference_adapt_step_19692iterator"
В
FullArgSpec
args

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z	capture_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
нBк
)__inference_embedding_layer_call_fn_21499inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
D__inference_embedding_layer_call_and_return_conditional_losses_21508inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ћBј
:__inference_global_average_pooling1d_4_layer_call_fn_21513inputs"Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
U__inference_global_average_pooling1d_4_layer_call_and_return_conditional_losses_21519inputs"Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
мBй
(__inference_dense_12_layer_call_fn_21528inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_dense_12_layer_call_and_return_conditional_losses_21539inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
мBй
(__inference_dense_13_layer_call_fn_21548inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_dense_13_layer_call_and_return_conditional_losses_21559inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
мBй
(__inference_dense_14_layer_call_fn_21568inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_dense_14_layer_call_and_return_conditional_losses_21579inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ьBщ
:__inference_transform_features_layer_1_layer_call_fn_20750text"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
:__inference_transform_features_layer_1_layer_call_fn_21584inputs/text"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
U__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_21605inputs/text"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
U__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_20776text"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ш
created_variables
	resources
trackable_objects
initializers
assets

signatures
$_self_saveable_object_factories
transform_fn"
_generic_user_object
0B.
__inference_pruned_20478inputsinputs_1
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
R
	variables
 	keras_api

Ёtotal

Ђcount"
_tf_keras_metric
c
Ѓ	variables
Є	keras_api

Ѕtotal

Іcount
Ї
_fn_kwargs"
_tf_keras_metric
"
_generic_user_object
Э
Јtrace_02Ў
__inference__creator_21610
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zЈtrace_0
б
Љtrace_02В
__inference__initializer_21618
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zЉtrace_0
Я
Њtrace_02А
__inference__destroyer_21623
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zЊtrace_0
Э
Ћtrace_02Ў
__inference__creator_21628
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zЋtrace_0
б
Ќtrace_02В
__inference__initializer_21633
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zЌtrace_0
Я
­trace_02А
__inference__destroyer_21638
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ z­trace_0
!J	
Const_2jtf.TrackableConstant
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
-
Ўserving_default"
signature_map
 "
trackable_dict_wrapper
0
Ё0
Ђ1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
Ѕ0
І1"
trackable_list_wrapper
.
Ѓ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
БBЎ
__inference__creator_21610"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ѕ
Џ	capture_1
А	capture_2BВ
__inference__initializer_21618"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zЏ	capture_1zА	capture_2
ГBА
__inference__destroyer_21623"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
БBЎ
__inference__creator_21628"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ЕBВ
__inference__initializer_21633"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ГBА
__inference__destroyer_21638"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
бBЮ
#__inference_signature_wrapper_20486inputsinputs_1"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
,:*	N2Adam/embedding/embeddings/m
&:$@2Adam/dense_12/kernel/m
 :@2Adam/dense_12/bias/m
&:$@ 2Adam/dense_13/kernel/m
 : 2Adam/dense_13/bias/m
&:$ 2Adam/dense_14/kernel/m
 :2Adam/dense_14/bias/m
,:*	N2Adam/embedding/embeddings/v
&:$@2Adam/dense_12/kernel/v
 :@2Adam/dense_12/bias/v
&:$@ 2Adam/dense_13/kernel/v
 : 2Adam/dense_13/bias/v
&:$ 2Adam/dense_14/kernel/v
 :2Adam/dense_14/bias/v
нBк
__inference_save_fn_21657checkpoint_key"Њ
В
FullArgSpec
args
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ	
 
B
__inference_restore_fn_21666restored_tensors_0restored_tensors_1"Е
В
FullArgSpec
args 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ
	
		6
__inference__creator_21610Ђ

Ђ 
Њ " 6
__inference__creator_21628Ђ

Ђ 
Њ " 8
__inference__destroyer_21623Ђ

Ђ 
Њ " 8
__inference__destroyer_21638Ђ

Ђ 
Њ " A
__inference__initializer_21618ZЏАЂ

Ђ 
Њ " :
__inference__initializer_21633Ђ

Ђ 
Њ " 
 __inference__wrapped_model_20709tZPQR*+23:;0Ђ-
&Ђ#
!
text_xfџџџџџџџџџ
Њ "3Њ0
.
dense_14"
dense_14џџџџџџџџџj
__inference_adapt_step_19692J[?Ђ<
5Ђ2
0-Ђ
џџџџџџџџџIteratorSpec 
Њ "
 Ѓ
C__inference_dense_12_layer_call_and_return_conditional_losses_21539\*+/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ@
 {
(__inference_dense_12_layer_call_fn_21528O*+/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ@Ѓ
C__inference_dense_13_layer_call_and_return_conditional_losses_21559\23/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ 
 {
(__inference_dense_13_layer_call_fn_21548O23/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ Ѓ
C__inference_dense_14_layer_call_and_return_conditional_losses_21579\:;/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 {
(__inference_dense_14_layer_call_fn_21568O:;/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџЇ
D__inference_embedding_layer_call_and_return_conditional_losses_21508_/Ђ,
%Ђ"
 
inputsџџџџџџџџџd	
Њ ")Ђ&

0џџџџџџџџџd
 
)__inference_embedding_layer_call_fn_21499R/Ђ,
%Ђ"
 
inputsџџџџџџџџџd	
Њ "џџџџџџџџџdд
U__inference_global_average_pooling1d_4_layer_call_and_return_conditional_losses_21519{IЂF
?Ђ<
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 Ќ
:__inference_global_average_pooling1d_4_layer_call_fn_21513nIЂF
?Ђ<
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
Њ "!џџџџџџџџџџџџџџџџџџД
B__inference_model_4_layer_call_and_return_conditional_losses_21192nZPQR*+23:;8Ђ5
.Ђ+
!
text_xfџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Д
B__inference_model_4_layer_call_and_return_conditional_losses_21266nZPQR*+23:;8Ђ5
.Ђ+
!
text_xfџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Г
B__inference_model_4_layer_call_and_return_conditional_losses_21409mZPQR*+23:;7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Г
B__inference_model_4_layer_call_and_return_conditional_losses_21492mZPQR*+23:;7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
'__inference_model_4_layer_call_fn_20925aZPQR*+23:;8Ђ5
.Ђ+
!
text_xfџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
'__inference_model_4_layer_call_fn_21118aZPQR*+23:;8Ђ5
.Ђ+
!
text_xfџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
'__inference_model_4_layer_call_fn_21299`ZPQR*+23:;7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
'__inference_model_4_layer_call_fn_21326`ZPQR*+23:;7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџє
__inference_pruned_20478зrЂo
hЂe
cЊ`
/
label&#
inputs/labelџџџџџџџџџ	
-
text%"
inputs/textџџџџџџџџџ
Њ "aЊ^
.
label_xf"
label_xfџџџџџџџџџ	
,
text_xf!
text_xfџџџџџџџџџy
__inference_restore_fn_21666Y[KЂH
AЂ>

restored_tensors_0

restored_tensors_1	
Њ " 
__inference_save_fn_21657і[&Ђ#
Ђ

checkpoint_key 
Њ "ШФ
`Њ]

name
0/name 
#

slice_spec
0/slice_spec 

tensor
0/tensor
`Њ]

name
1/name 
#

slice_spec
1/slice_spec 

tensor
1/tensor	і
#__inference_signature_wrapper_20486ЮiЂf
Ђ 
_Њ\
*
inputs 
inputsџџџџџџџџџ	
.
inputs_1"
inputs_1џџџџџџџџџ"aЊ^
.
label_xf"
label_xfџџџџџџџџџ	
,
text_xf!
text_xfџџџџџџџџџЄ
#__inference_signature_wrapper_20625}ZPQR*+23:;9Ђ6
Ђ 
/Њ,
*
examples
examplesџџџџџџџџџ"3Њ0
.
output_0"
output_0џџџџџџџџџд
U__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_20776{:Ђ7
0Ђ-
+Њ(
&
text
textџџџџџџџџџ
Њ "=Ђ:
3Њ0
.
text_xf# 
	0/text_xfџџџџџџџџџ
 м
U__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_21605AЂ>
7Ђ4
2Њ/
-
text%"
inputs/textџџџџџџџџџ
Њ "=Ђ:
3Њ0
.
text_xf# 
	0/text_xfџџџџџџџџџ
 ­
:__inference_transform_features_layer_1_layer_call_fn_20750o:Ђ7
0Ђ-
+Њ(
&
text
textџџџџџџџџџ
Њ "1Њ.
,
text_xf!
text_xfџџџџџџџџџД
:__inference_transform_features_layer_1_layer_call_fn_21584vAЂ>
7Ђ4
2Њ/
-
text%"
inputs/textџџџџџџџџџ
Њ "1Њ.
,
text_xf!
text_xfџџџџџџџџџ